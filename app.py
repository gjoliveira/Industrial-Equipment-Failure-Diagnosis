import streamlit as st
import pandas as pd
import altair as alt
import rdflib
import sys
import os
import pickle
import numpy as np
import time

# --- Configuration ---
# Set up the Streamlit page configuration
st.set_page_config(
    page_title="Industrial Diagnosis System",
    page_icon="🏭",
    layout="wide"
)

# Constants defining file paths and default costs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(BASE_DIR, "ontologies","cnc_final.owl")
DATA_PATH = os.path.join(BASE_DIR, "data", "telemetry.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")


# Add 'ForTheApp' directory to sys.path to allow importing bn_codes
sys.path.append(os.path.join(BASE_DIR, "ForTheApp"))

# Attempt to import the Bayesian Network helper module
try:
    import ForTheApp.bn_codes as bn_codes
except ImportError as e:
    st.error(f"Error importing bn_codes: {e}")
    st.stop()

# --- Helper Functions ---

@st.cache_resource
def load_ontology():
    """Loads the OWL ontology file."""
    g = rdflib.Graph()
    try:
        g.parse(ONTOLOGY_PATH)
        return g
    except Exception as e:
        st.error(f"Error loading ontology: {e}")
        return None

@st.cache_resource
def load_bn_model(model_name):
    """Loads the Bayesian Network model from a pickle file."""
    model_path = os.path.join(MODELS_DIR, model_name)
    try:
        # Load the model using the helper or standard pickle
        model = bn_codes.load_model(model_path, check=False)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

def validate_model(model):
    """
    Validates if the BN model contains the necessary nodes for this application.
    Required: 'Overheat' node and at least one latent cause (e.g., BearingWear, FanFault).
    """
    if not hasattr(model, "nodes"):
        return False, "Model object does not have 'nodes' method."
        
    nodes = set(model.nodes())
    required_target = "Overheat"
    latent_causes = ["BearingWear", "FanFault", "CloggedFilter", "CoolingEfficiency"]
    
    if required_target not in nodes:
        return False, f"Model is missing the target node: '{required_target}'"
        
    # Check if ANY latent cause is present in the model
    has_latent = any(cause in nodes for cause in latent_causes)
    
    if not has_latent:
        return True, "Model valid for Overheat prediction only (No root cause analysis)."
        
    return True, "Model is valid (Predicts Overheat + Root Causes)."

@st.cache_data
def load_data():
    try:
        telemetry = pd.read_csv(DATA_PATH)
        return telemetry
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def process_evidence(inputs):
    """
    Prepares evidence dictionary for the Bayesian Network.
    - Numeric inputs are discretized using helper functions.
    - Categorical inputs are passed directly.
    """
    evidence = {}
    
    # Retrieve inputs
    val_temp = inputs.get('spindle_temp')
    val_vib = inputs.get('vibration_rms')
    val_flow = inputs.get('coolant_flow')

    # Process Spindle Temperature
    if isinstance(val_temp, (int, float, np.number)):
        evidence['SpindleTemp'] = bn_codes.spindle_temp_bin(val_temp)
    else:
        evidence['SpindleTemp'] = val_temp

    # Process Vibration
    if isinstance(val_vib, (int, float, np.number)):
        evidence['Vibration'] = bn_codes.vibration_rms_bin(val_vib)
    else:
        evidence['Vibration'] = val_vib

    # Process Coolant Flow
    if isinstance(val_flow, (int, float, np.number)):
        evidence['CoolantFlow'] = bn_codes.coolant_flow_bin(val_flow)
    else:
        evidence['CoolantFlow'] = val_flow
        
    return evidence

def get_kg_recommendation(graph, cause_name):
    """
    Queries the Knowledge Graph (Ontology) for maintenance actions related to a specific cause.
    """
    if not graph:
        return None

    # SPARQL Query to find actions that mitigate the inferred failure mode
    query = f"""
    PREFIX gjoli: <http://example.org/gjoli_diagnostics#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    
    SELECT ?action ?actionName ?cost ?duration ?desc
    WHERE {{
        ?action a gjoli:MaintenanceAction ;
                gjoli:mitigates ?failureMode .
        
        # Filter to match the failure mode name with the cause name (fuzzy match)
        FILTER(REGEX(STR(?failureMode), "{cause_name}", "i"))
        
        OPTIONAL {{ ?action gjoli:sparePartsCost ?cost . }}
        OPTIONAL {{ ?action gjoli:hasDurationHours ?duration . }}
        BIND(STRAFTER(STR(?action), "#") AS ?actionName)
        BIND("Detected " + "{cause_name}" + ". Action required." AS ?desc)
    }}
    LIMIT 1
    """
    
    results = graph.query(query)
    
    rec = {}
    for row in results:
        rec['action_name'] = str(row.actionName)
        rec['cost'] = float(row.cost) if row.cost else 0.0
        rec['duration'] = float(row.duration) if row.duration else 0.0
        rec['description'] = str(row.desc)
        return rec
    
    return None



def update_row_index():
    """Callback to synchronize the row index state with the widget."""
    st.session_state.row_index = st.session_state.widget_row_index

# --- Main App Logic ---

def main():
    st.title("⚙️ Intelligent Industrial Diagnosis System")
    st.markdown("### Hybrid BN + KG Diagnosis Dashboard")
    
    # 1. Sidebar Configuration
    st.sidebar.header("Configuration")
    
    # Select Bayesian Network Model
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    selected_model_name = st.sidebar.selectbox("Select BN Model", model_files)
    
    # Strategy Settings Group
    with st.sidebar.expander("🛠️ Strategy Settings", expanded=False):
        hourly_downtime_cost = st.number_input("Hourly Downtime Cost (€/h)", value=80.0)
        failure_cost_default = st.number_input("Default Failure Cost (€)", value=2000.0)
        slowdown_factor = st.slider("Slowdown Risk Factor", 0.0, 1.0, 0.5, help="Risk reduction when slowing down")
        slowdown_penalty = st.number_input("Slowdown Penalty (€/h)", value=40.0, help="Productivity loss per hour")
    
    # Data Input Mode Selection
    st.sidebar.divider()
    input_mode = st.sidebar.radio("Input Mode", ["Manual Input", "Historical Data (CSV)"], horizontal=True)
    
    # Load external resources (Ontology and BN Model)
    graph = load_ontology()
    bn_model = load_bn_model(selected_model_name)
    
    # Validate the loaded model's capabilities
    model_has_latents = False
    
    if bn_model:
        is_valid, msg = validate_model(bn_model)
        if not is_valid:
            st.error(f"⚠️ Invalid Model Selected: {msg}")
            bn_model = None
        else:
            if "No root cause" in msg:
                 st.caption(f"⚠️ {msg}")
                 model_has_latents = False
            else:
                 model_has_latents = True
            
            
    telemetry_df = load_data()
    
    inputs = {}
    
    if input_mode == "Manual Input":
        # Mode: User manually adjusts sliders/dropdowns
        st.sidebar.subheader("Sensor Inputs")
        
        input_type = st.sidebar.radio("Input Format", ["Values (Numeric)", "Categories (Discrete)"])
        
        if input_type == "Values (Numeric)":
            t = st.sidebar.slider("Spindle Temperature (°C)", 0.0, 120.0, 60.0)
            v = st.sidebar.slider("Vibration RMS (mm/s)", 0.0, 5.0, 0.8)
            c = st.sidebar.slider("Coolant Flow (L/min)", 0.0, 5.0, 1.0)
        else:
            # Dropdowns for discrete states
            t = st.sidebar.selectbox("Spindle Temperature", ["low", "moderate", "high"], index=1)
            v = st.sidebar.selectbox("Vibration RMS", ["low", "moderate", "high"], index=1)
            c = st.sidebar.selectbox("Coolant Flow", ["low", "moderate", "high"], index=1)
        
        inputs = {
            'spindle_temp': t,
            'vibration_rms': v,
            'coolant_flow': c
        }
        
        # Display current values in the main area
        with st.container(border=True):
            st.markdown("#### Values")
            st.caption("Adjust sensors above")
            c1, c2, c3 = st.columns(3)
            c1.metric("Temp", t)
            c2.metric("Vib", v)
            c3.metric("Flow", c)
        
    else:
        # Mode: Historical Data from CSV
        st.sidebar.subheader("History Selection")
        
        if telemetry_df is not None:
            max_idx = len(telemetry_df) - 1
            
            # --- Simulation Controls ---
            if 'row_index' not in st.session_state:
                st.session_state.row_index = 0
            if 'sim_active' not in st.session_state:
                st.session_state.sim_active = False
                
            # Synchronize widget state with simulation state
            if 'widget_row_index' not in st.session_state:
                st.session_state.widget_row_index = st.session_state.row_index
            elif st.session_state.widget_row_index != st.session_state.row_index:
                st.session_state.widget_row_index = st.session_state.row_index
                
            st.sidebar.markdown("### ⏯️ Simulation")
            col_p, col_s = st.sidebar.columns(2)
            
            if col_p.button("▶️ Play"):
                st.session_state.sim_active = True
                st.rerun()
                
            if col_s.button("⏹️ Stop"):
                st.session_state.sim_active = False
                st.rerun()
                
            st.sidebar.radio("Speed", ["2s", "5s"], key="sim_speed_opt", horizontal=True)
            
            # --- Row Selection Widget ---
            # Ensure index is within bounds
            if st.session_state.row_index > max_idx:
                 st.session_state.row_index = 0
                 
            # Note: Usage of callback to ensure 2-way binding
            idx = st.sidebar.number_input(
                f"Select Row Index (0-{max_idx})", 
                0, max_idx, 
                key="widget_row_index",
                on_change=update_row_index
            )
            
            row = telemetry_df.iloc[idx]
            st.sidebar.markdown(f"**Timestamp**: {row['timestamp']}")
            st.sidebar.markdown(f"**Machine**: {row['machine_id']}")
            
            inputs = {
                'spindle_temp': row['spindle_temp'],
                'vibration_rms': row['vibration_rms'],
                'coolant_flow': row['coolant_flow']
            }
            
            # Show the selected parameters
            d_t = bn_codes.spindle_temp_bin(inputs['spindle_temp'])
            d_v = bn_codes.vibration_rms_bin(inputs['vibration_rms'])
            d_c = bn_codes.coolant_flow_bin(inputs['coolant_flow'])

            st.sidebar.info(f"""
            **Current Values:**
            - Temp: {inputs['spindle_temp']:.2f} ({d_t})
            - Vib: {inputs['vibration_rms']:.2f} ({d_v})
            - Flow: {inputs['coolant_flow']:.2f} ({d_c})
            """)
    
    # 2. Diagnosis Section (Vertical Layout)
    
    # --- Section 1: Bayesian Diagnosis ---
    st.markdown("---")
    st.subheader("1. Bayesian Diagnosis")
    
    bn_ranking = []
    p_risk = 0.0
    
    if bn_model:
        # Prepare evidence for the BN
        evidence = process_evidence(inputs)
        
        # Display inputs and evidence
        with st.expander("🔍 Inference Details (Evidence)", expanded=False):
            st.write("Processed Evidence:", evidence)
            
        try:
            # Perform diagnosis
            if model_has_latents:
                # Full diagnosis with root causes
                p_risk, bn_ranking = bn_codes.diagnose_overheat(bn_model, evidence)
            else:
                # Simple prediction (target only)
                p_risk = bn_codes.overheat_probability(bn_model, evidence)
                bn_ranking = []
            
            # Display Probability and Root Cause Chart
            m1, m2 = st.columns([1, 2])
            
            with m1:
                 st.metric("🔥 Overheat Probability", f"{p_risk:.2%}", delta_color="inverse")
                 
            with m2:
                if bn_ranking:
                    # Altair Bar Chart for Root Causes
                    chart_data = pd.DataFrame(bn_ranking, columns=["Cause", "Probability"])
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X('Probability', axis=alt.Axis(format='%')),
                        y=alt.Y('Cause', sort='-x'),
                        color=alt.Color('Probability', scale=alt.Scale(scheme='orangered'))
                    ).properties(height=200)
                    st.altair_chart(chart, use_container_width=True)
                elif not model_has_latents:
                     st.info("No Root Cause Analysis available using simple MLE model.")
                else:
                    st.info("Risk is low. No latent causes to display.")
                    
        except Exception as e:
            st.error(f"Inference Error: {e}")
    else:
        st.warning("Model not loaded.")

    # --- Section 2: Decision Support ---
    st.markdown("---") 
    st.subheader("2. Decision Support (KG)")
    
    ranking = bn_ranking
    
    if not ranking:
        if p_risk < 0.5:
             st.success("✅ System Status Nominal. Monitoring continues.")
        else:
             st.warning("⚠️ Overheat Risk Detectable, but Root Cause Unclear. Manual Inspection Advised.")
    else:
            # Strategy: Plan A (Max Priority/Probability)
            # Logic: Identify the most likely failure, consult KG for repair actions, 
            # and calculate the Expected Total Cost for different decisions.
            
            probabilities = {cause: prob for cause, prob in ranking}
            ranked_options = []
            
            # 1. Identify failure with highest probability
            cause_name, probability = ranking[0]
            
            # 2. Retrieve repair details from the Knowledge Graph
            details = get_kg_recommendation(graph, cause_name)
            
            if details:
                # 3. Calculate Expected Costs for each Decision Option
                # Cost = Probability * Impact
                
                # Case 1: Continue Operation (Risk of Failure)
                # Cost is the expected failure cost if failure occurs
                cost_continue = probability * failure_cost_default
                
                # Case 2: Slow Down (Risk Mitigation)
                # Reduced failure risk (slowdown_factor) + Cost of lost productivity (penalty)
                cost_slow = (probability * slowdown_factor * failure_cost_default) + (slowdown_penalty)
                
                # Case 3: Maintenance (Immediate Repair)
                # Cost is Parts + (Duration * Hourly Downtime)
                cost_maint = details['cost'] + (details['duration'] * hourly_downtime_cost)
                
                options = [
                    ("Continue", cost_continue),
                    ("Slow down", cost_slow),
                    ("Schedule maintenance", cost_maint)
                ]
                
                # Select the decision with the Minimum Expected Cost
                best_decision, best_cost = min(options, key=lambda x: x[1])
                
                ranked_options.append({
                    "failure": cause_name,
                    "prob": probability,
                    "decision": best_decision,
                    "expected_cost": best_cost,
                    "details": details,
                    "options": options
                })

            # Display Recommended Code
            if ranked_options:
                top_opt = ranked_options[0]
                
                st.markdown(f"### 💡 Recommended: **{top_opt['decision']}**")
                st.markdown(f"**Focus on:** `{top_opt['failure']}` (Prob: {top_opt['prob']:.1%})")
                st.metric("Expected Total Cost", f"€{top_opt['expected_cost']:.0f}")
                
                st.divider()
                st.markdown("#### 📊 Detailed Analysis")
                
                # Visual breakdown of costs for the recommended option
                opts = top_opt['options']
                cols = st.columns(3)
                for i, (label, val) in enumerate(opts):
                    cols[i].metric(label, f"€{val:.0f}", delta=None)
                
                # Show Action Details if Maintenance is the recommendation
                if "maintenance" in top_opt['decision'].lower():
                    st.divider()
                    details = top_opt['details']
                    st.info(f"**Procedure**: {details['action_name']} ({details['duration']}h, €{details['cost']})")
                    
            else:
                st.warning("Could not determine optimal decision (missing KG data?).")

    # --- Simulation Loop Logic ---
    # This block handles the automatic iteration when 'Play' is active.
    # It renders the page, waits, increments the index, and forces a rerun.
    if input_mode == "Historical Data (CSV)" and st.session_state.get('sim_active'):
        speed_val = st.session_state.get("sim_speed_opt", "2s")
        delay = 2 if speed_val == "2s" else 5
        
        # Verify bounds before incrementing
        if telemetry_df is not None:
            max_rows = len(telemetry_df) - 1
            if st.session_state.row_index < max_rows:
                time.sleep(delay)
                st.session_state.row_index += 1
                st.rerun()
            else:
                st.session_state.sim_active = False
                st.success("Simulation Complete.")

if __name__ == "__main__":
    main()
