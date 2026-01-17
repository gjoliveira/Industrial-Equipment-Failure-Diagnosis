import numpy as np
import pickle
from typing import Any
from pgmpy.inference import VariableElimination

# ---------------------------------------------
# --------------- DISCRETIZATION --------------
# ---------------------------------------------


"""
    Transforms a number x into a category: 
    - "low" if x <= 65, 
    - "moderate" if 65 < x <= 81,
    - "high" if x > 81

    (-inf, 65] -> low | (65, 81] -> moderate | (81, inf) -> high
"""
def spindle_temp_bin(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x <= 65:
        return "low"
    elif x <= 81:
        return "moderate"
    else:
        return "high"

# ---------------------------------------------

"""
    Transforms a number x into a category: 
    - "low" if x <= 0.90, 
    - "moderate" if 0.90 < x <= 1.32,
    - "high" if x > 1.32

    (-inf, 0.90] -> low | (0.90, 1.32] -> moderate | (1.32, inf) -> high
"""
def vibration_rms_bin(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x <= 0.90:
        return "low"
    elif x <= 1.32:
        return "moderate"
    else:
        return "high"

# ---------------------------------------------

"""
    Transforms a number x into a category: 
    - "low" if x <= 0.45, 
    - "moderate" if 0.45 < x <= 0.80,
    - "high" if x > 0.80

    [-inf, 0.45) -> low | [0.45, 0.80) -> moderate | [0.80, inf) -> high
"""
def coolant_flow_bin(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if x < 0.45:
        return "low"
    elif x < 0.80:
        return "moderate"
    else:
        return "high"

# ---------------------------------------------
# --------------- LOAD BN MODEL ---------------
# ---------------------------------------------

"""
    Loads a Bayesian Network model from a pickle file.
    We need to give to it the path where the model is stored.
    If `check` is True, and the loaded object has a method `check_model()`, it will be executed.
"""
def load_model(model_path: str, check: bool = True, debug: bool = False) -> Any:
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if check and hasattr(model, "check_model") and callable(getattr(model, "check_model")):
        model.check_model()
        if debug:
            print("Model check passed.")
        

    return model

# ---------------------------------------------
# ------ GET OVERHEAT PROBABILITY ON MLE ------
# ---------------------------------------------



"""
    Calculates P(Overheat = overheat_state | evidence).
    - evidence: dict with sensors (e.g., {"SpindleTemp":2, "Vibration":1, "CoolantFlow":0}).
    - overheat_state: state of interest for Overheat node (usually 1).
"""
def overheat_probability(model, evidence, overheat_state=1) -> float:
    
    infer = VariableElimination(model)
    q = infer.query(variables=["Overheat"], evidence=evidence, show_progress=False)

    states = q.state_names["Overheat"]  # list of states in the same order as q.values

    # try direct match
    if overheat_state in states:
        idx = states.index(overheat_state)
    else:
        # try match by string representation
        ps = str(overheat_state)
        idx = next((i for i, s in enumerate(states) if str(s) == ps), None)

        # fallback: if binary and not found, assume "positive" is index 1
        if idx is None:
            if len(states) == 2:
                idx = 1
            else:
                raise ValueError(
                    f"State {overheat_state} not found in Overheat. Available states: {states}"
                )

    return float(q.values[idx])

# ---------------------------------------------
# ---------- GET OVERHEAT PROBABILITY ---------
# ------------ AND DIAGNOSIS ON ME -----------
# ---------------------------------------------

"""
    Calculates P(Overheat = 1 | evidence).
    Ranks the possible latent causes of overheat given the evidence.

    - ranking: list[(latent_name, p_fault)] sorted desc, where
        p_fault = P(latent=latent_fault_state | evidence, overheat=1)
      If p_overheat < 0.5 => ranking is [] (empty)
"""
def diagnose_overheat(model, evidence: dict, latent_fault_state=1, top_k: int | None = 4) -> tuple[float, list[tuple[str, float]]]:
    
    # --- P(Overheat | sensors)
    p_overheat = overheat_probability(model, evidence)

    # --- Ranking of latent causes if above threshold
    ranking = []
    if p_overheat >= 0.5:   # threshold
        infer = VariableElimination(model)
        ev = dict(evidence) # copy to not modify original
        ev["Overheat"] = 1  # condition on overheat happening

        for lv in ["BearingWear", "FanFault", "CloggedFilter", "CoolingEfficiency"]:
            q = infer.query(variables=[lv], evidence=ev, show_progress=False)
            states_lv = q.state_names[lv]

            if latent_fault_state in states_lv:
                idx_lv = states_lv.index(latent_fault_state)
            else:
                lfs_ = str(latent_fault_state)
                idx_lv = next((i for i, s in enumerate(states_lv) if str(s) == lfs_), None)
                if idx_lv is None:
                    if len(states_lv) == 2:
                        idx_lv = 1
                    else:
                        raise ValueError(
                            f"State {latent_fault_state} not found in {lv}. Available states: {states_lv}"
                        )

            ranking.append((lv, float(q.values[idx_lv])))

        ranking.sort(key=lambda x: x[1], reverse=True)

    return p_overheat, ranking

# ---------------------------------------------

"""
    Prints the diagnosis results:
    - evidence sensors
    - P(Overheat=1 | evidence)
    - ranking of latent causes (if p_overheat >= 0.5)
"""
def print_diagnosis(p_overheat: float, ranking: list[tuple[str, float]], evidence: dict, top_k: int | None = 4) -> None:
    print("Evidence (sensors):", dict(evidence))
    print(f"P(Overheat=1 | evidence) = {p_overheat:.4f}")

    if p_overheat < 0.5:
        print(f"Without cause ranking (p<0.5).")
    elif not ranking:
        print("No latents provided (ranking empty).")
    else:
        print("\nRanking of latent causes (conditioned on Overheat=1):")
        shown = ranking if top_k is None else ranking[:top_k]
        for lv, p in shown:
            print(f"  - {lv}: P(fault) = {p:.4f}")
