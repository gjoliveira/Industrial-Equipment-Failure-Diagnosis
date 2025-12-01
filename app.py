import streamlit as st
import pandas as pd
import numpy as np
import rdflib
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
import networkx as nx
import matplotlib.pyplot as plt
import time

# Page Config
st.set_page_config(
    page_title="Industrial Diagnostics System",
    page_icon="🏭",
    layout="wide"
)

# Title and Description
st.title("🏭 Hybrid Neuro-Symbolic Industrial Diagnostics")
st.markdown("""
This system combines **Probabilistic Reasoning (Bayesian Networks)** with **Semantic Knowledge (Knowledge Graphs)** 
to diagnose failures in CNC milling machines and recommend maintenance actions.
""")

# --- 1. Load Knowledge Graph ---
@st.cache_resource
def load_knowledge_graph():
    g = rdflib.Graph()
    try:
        g.parse("knowledge_base.ttl", format="turtle")
        return g
    except Exception as e:
        st.error(f"Failed to load Knowledge Graph: {e}")
        return None

g = load_knowledge_graph()

if g:
    st.sidebar.success(f"Knowledge Graph Loaded ({len(g)} triples)")

# --- 2. Initialize Bayesian Network ---
@st.cache_resource
def load_bayesian_network():
    # Define the model structure
    model = DiscreteBayesianNetwork([
        ('BearingWearHigh', 'vibration_rms'),
        ('FanFault', 'spindle_temp'),
        ('CloggedFilter', 'coolant_flow'),
        ('LowCoolingEfficiency', 'spindle_temp')
    ])

    # Initialize CPDs (Expert Knowledge / Randomized for Prototype)
    
    # Latent Nodes (Roots)
    cpd_bearing = TabularCPD(variable='BearingWearHigh', variable_card=2, values=[[0.5], [0.5]], state_names={'BearingWearHigh': [0, 1]})
    cpd_fan = TabularCPD(variable='FanFault', variable_card=2, values=[[0.5], [0.5]], state_names={'FanFault': [0, 1]})
    cpd_filter = TabularCPD(variable='CloggedFilter', variable_card=2, values=[[0.5], [0.5]], state_names={'CloggedFilter': [0, 1]})
    cpd_cooling = TabularCPD(variable='LowCoolingEfficiency', variable_card=2, values=[[0.5], [0.5]], state_names={'LowCoolingEfficiency': [0, 1]})

    # Observed Nodes (Children)
    # We define states: High, Low, Medium
    states = ['High', 'Low', 'Medium']
    
    def get_random_cpd(var, card, evidence, evidence_card, states, evidence_states):
        values = np.random.rand(card, np.prod(evidence_card))
        values = values / values.sum(axis=0)
        return TabularCPD(variable=var, variable_card=card, values=values, 
                          evidence=evidence, evidence_card=evidence_card,
                          state_names={var: states, evidence[0]: evidence_states if evidence else []})

    cpd_vib = get_random_cpd('vibration_rms', 3, ['BearingWearHigh'], [2], states, [0, 1])
    cpd_coolant = get_random_cpd('coolant_flow', 3, ['CloggedFilter'], [2], states, [0, 1])

    # spindle_temp has 2 parents
    values_temp = np.random.rand(3, 2*2)
    values_temp = values_temp / values_temp.sum(axis=0)
    cpd_temp = TabularCPD(variable='spindle_temp', variable_card=3, 
                          values=values_temp,
                          evidence=['FanFault', 'LowCoolingEfficiency'], evidence_card=[2, 2],
                          state_names={'spindle_temp': states, 'FanFault': [0, 1], 'LowCoolingEfficiency': [0, 1]})

    model.add_cpds(cpd_bearing, cpd_fan, cpd_filter, cpd_cooling, cpd_vib, cpd_coolant, cpd_temp)
    
    if model.check_model():
        return model
    else:
        st.error("Bayesian Network Model Check Failed!")
        return None

model = load_bayesian_network()
infer = None
if model:
    infer = VariableElimination(model)
    st.sidebar.success("Bayesian Network Initialized")

# --- 3. Data Loading & Processing for Simulation ---
@st.cache_data
def load_telemetry_data():
    try:
        df = pd.read_csv('data/telemetry.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"Failed to load telemetry data: {e}")
        return None

def discretize_value(val, thresholds):
    if val <= thresholds[0]:
        return 'Low'
    elif val <= thresholds[1]:
        return 'Medium'
    else:
        return 'High'

def get_thresholds(df, column):
    # Simple quantile-based thresholds (33%, 66%)
    return df[column].quantile([0.33, 0.66]).values

# --- 4. Helper Functions ---

def run_diagnostics(evidence, key_suffix=""):
    if not infer:
        st.error("Inference engine not initialized.")
        return

    st.subheader("Diagnostic Results")
    
    # 1. BN Inference
    failures = ['BearingWearHigh', 'FanFault', 'CloggedFilter', 'LowCoolingEfficiency']
    results = []
    
    for failure in failures:
        try:
            q = infer.query(variables=[failure], evidence=evidence)
            prob = q.values[1] # Probability of True
            results.append({'failure': failure, 'probability': prob})
        except Exception as e:
            # st.error(f"Inference error for {failure}: {e}")
            pass

    if not results:
        st.warning("No inference results.")
        return

    # Display Probabilities
    df_res = pd.DataFrame(results).sort_values(by='probability', ascending=False)
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.write("### Failure Probabilities")
        st.dataframe(df_res.style.format({'probability': "{:.2%}"}))
        
        # Plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.barh(df_res['failure'], df_res['probability'], color='skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

    # 2. KG Recommendation
    with c2:
        st.write("### Recommended Actions")
        
        top_failure = df_res.iloc[0]['failure']
        top_prob = df_res.iloc[0]['probability']
        
        st.info(f"Most Likely Cause: **{top_failure}** ({top_prob:.2%})")
        
        # Query KG
        if g:
            query = f"""
                PREFIX cnc: <http://example.org/cnc#>
                SELECT ?action ?cost ?effort
                WHERE {{
                    ?action cnc:mitigates cnc:{top_failure} .
                    ?action cnc:cost ?cost .
                    ?action cnc:effort ?effort .
                }}
            """
            
            actions = []
            for row in g.query(query):
                action_name = row.action.split('#')[1]
                actions.append({
                    'Action': action_name,
                    'Cost (€)': float(row.cost),
                    'Effort (h)': float(row.effort)
                })
            
            if actions:
                st.table(pd.DataFrame(actions))
            else:
                st.warning("No specific maintenance action found in Knowledge Graph for this failure.")
        else:
            st.warning("Knowledge Graph not loaded.")

# --- 5. UI Pages ---

def render_manual_diagnostics():
    st.header("🔍 Manual Diagnostic Interface")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        vib_input = st.selectbox("Vibration RMS", ['Low', 'Medium', 'High'], index=1)
    with col2:
        temp_input = st.selectbox("Spindle Temperature", ['Low', 'Medium', 'High'], index=1)
    with col3:
        coolant_input = st.selectbox("Coolant Flow", ['Low', 'Medium', 'High'], index=1)

    evidence = {
        'vibration_rms': vib_input,
        'spindle_temp': temp_input,
        'coolant_flow': coolant_input
    }

    if st.button("Run Diagnostics"):
        run_diagnostics(evidence)

def render_simulation_mode():
    st.header("🔄 Machine Simulation Mode")
    
    df = load_telemetry_data()
    if df is None:
        return

    # Calculate thresholds for discretization
    # We need thresholds for: vibration_rms, spindle_temp, coolant_flow
    # Assuming these columns exist in telemetry.csv. 
    req_cols = ['vibration_rms', 'spindle_temp', 'coolant_flow']
    if not all(col in df.columns for col in req_cols):
        st.error(f"Telemetry data missing required columns: {req_cols}")
        st.write("Available columns:", df.columns.tolist())
        return

    thresh_vib = get_thresholds(df, 'vibration_rms')
    thresh_temp = get_thresholds(df, 'spindle_temp')
    thresh_coolant = get_thresholds(df, 'coolant_flow')

    st.sidebar.markdown("### Simulation Controls")
    machine_id = st.sidebar.selectbox("Select Machine", df['machine_id'].unique())
    
    df_machine = df[df['machine_id'] == machine_id].sort_values('timestamp')
    
    start_idx = st.sidebar.slider("Start Index", 0, len(df_machine)-1, 0)
    speed = st.sidebar.slider("Simulation Speed (sec/step)", 0.1, 2.0, 0.5)
    
    if st.sidebar.button("Start Simulation"):
        st_frame = st.empty()
        
        for i in range(start_idx, len(df_machine)):
            row = df_machine.iloc[i]
            
            # Discretize
            vib_d = discretize_value(row['vibration_rms'], thresh_vib)
            temp_d = discretize_value(row['spindle_temp'], thresh_temp)
            coolant_d = discretize_value(row['coolant_flow'], thresh_coolant)
            
            evidence = {
                'vibration_rms': vib_d,
                'spindle_temp': temp_d,
                'coolant_flow': coolant_d
            }
            
            with st_frame.container():
                st.subheader(f"Time: {row['timestamp']}")
                
                # Metrics
                c1, c2, c3 = st.columns(3)
                c1.metric("Vibration", f"{row['vibration_rms']:.4f}", vib_d)
                c2.metric("Temperature", f"{row['spindle_temp']:.2f}", temp_d)
                c3.metric("Coolant Flow", f"{row['coolant_flow']:.2f}", coolant_d)
                
                st.markdown("---")
                run_diagnostics(evidence, key_suffix=f"_{i}")
                
            time.sleep(speed)

def render_knowledge_graph():
    st.header("🕸️ Knowledge Graph Structure")
    if g:
        # Enhanced Visualization
        sg = nx.DiGraph()
        
        # Query triples with types for coloring
        query_viz = """
            PREFIX cnc: <http://example.org/cnc#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?s ?p ?o ?sType ?oType
            WHERE {
                ?s ?p ?o .
                OPTIONAL { ?s rdf:type ?sType } .
                OPTIONAL { ?o rdf:type ?oType } .
                FILTER (?p = cnc:manifestsAs || ?p = cnc:mitigates)
            }
            LIMIT 50
        """
        
        node_colors = []
        node_types = {}
        
        for row in g.query(query_viz):
            s = row.s.split('#')[1] if '#' in row.s else row.s
            o = row.o.split('#')[1] if '#' in row.o else row.o
            p = row.p.split('#')[1] if '#' in row.p else row.p
            
            sg.add_edge(s, o, label=p)
            
            # Determine types for coloring
            if row.sType:
                node_types[s] = row.sType.split('#')[1]
            if row.oType:
                node_types[o] = row.oType.split('#')[1]

        # Define colors
        color_map = {
            'FailureMode': '#ff9999',       # Red
            'Symptom': '#ffcc99',           # Orange
            'MaintenanceAction': '#99ff99', # Green
            'Component': '#99ccff',         # Blue
            'default': '#e0e0e0'            # Grey
        }
        
        colors = [color_map.get(node_types.get(node, 'default'), color_map['default']) for node in sg.nodes()]
        
        fig_net, ax_net = plt.subplots(figsize=(14, 10))
        # Increase k to separate nodes more (default is 1/sqrt(n))
        pos = nx.spring_layout(sg, k=2.5, iterations=100, seed=42)
        
        nx.draw(sg, pos, with_labels=True, node_color=colors, node_size=2500, font_size=9, font_weight='bold', edge_color='gray', arrows=True, ax=ax_net)
        edge_labels = nx.get_edge_attributes(sg, 'label')
        nx.draw_networkx_edge_labels(sg, pos, edge_labels=edge_labels, font_color='red', ax=ax_net)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', label='Failure Mode', markerfacecolor=color_map['FailureMode'], markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Symptom', markerfacecolor=color_map['Symptom'], markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Action', markerfacecolor=color_map['MaintenanceAction'], markersize=15),
            plt.Line2D([0], [0], marker='o', color='w', label='Component', markerfacecolor=color_map['Component'], markersize=15)
        ]
        ax_net.legend(handles=legend_elements, loc='upper right')
        
        st.pyplot(fig_net)

# --- Main App Logic ---
page = st.sidebar.radio("Navigation", ["Manual Diagnostics", "Simulation Mode", "Knowledge Graph"])

if page == "Manual Diagnostics":
    render_manual_diagnostics()
elif page == "Simulation Mode":
    render_simulation_mode()
elif page == "Knowledge Graph":
    render_knowledge_graph()
