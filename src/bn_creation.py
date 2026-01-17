"""
This file has all the functions used during the data analysis for the Bayesian Network.
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score, accuracy_score, 
                             precision_score, recall_score, f1_score)
from pgmpy.estimators import ExpectationMaximization
from IPython.display import display, Markdown

# ---------------------------------------------------------------

"""
    Creates the discrete model baseline (only obervables) with edges:
        features -> target

    Exemple:
        SpindleTemp -> Overheat
        Vibration   -> Overheat
        CoolantFlow -> Overheat
"""
def build_mle_model(features, target):
    edges = [(f, target) for f in features]
    return DiscreteBayesianNetwork(edges)

# ---------------------------------------------------------------

"""
    Learns the CPDs of the model using Maximum Likelihood Estimation (MLE).
    It does counts on train_df to estimate P(node | parents).

    IMPORTANT:
      - The train_df must have DISCRETE variables (integers/categories).
      - There should be no NaNs in the used columns.
"""
def fit_mle(model, train_df, features, target):
    cols = features + [target]
    model.fit(train_df[cols], estimator=MaximumLikelihoodEstimator)
    model.check_model()  # validates if the CPDs match the structure
    return model

# ---------------------------------------------------------------

"""
    Prints a full CPD (TabularCPD) without truncating the table.
    Does a temporary monkey-patch and restores it at the end.
"""
def print_full_cpd(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, s: s
    try:
        print(cpd)
    finally:
        TabularCPD._truncate_strtable = backup

# ---------------------------------------------------------------

"""
    Prints all learned CPDs of the model.
    Useful for including in reports: shows P(SpindleTemp), P(Vibration), ... and P(Overheat|sensors).
"""
def show_model_cpds(model):    
    for cpd in model.get_cpds():
        print_full_cpd(cpd)
        print("\n")

# ---------------------------------------------------------------

"""
    Given the result of a query (DiscreteFactor), finds the index of the desired state.

    This makes the code robust because:
      - sometimes the states are strings ("0","1","yes","no")
      - or integers (0,1)
      - and the order may not be what we assume

    Returns the index where the desired_state is located.
"""
def state_index(query_factor, var_name, desired_state):
    states = query_factor.state_names[var_name]  # list of states (in the order of values)

    # attempt 1: direct match
    if desired_state in states:
        return states.index(desired_state)

    # attempt 2: match by string (e.g., desired_state=1, states=["0","1"])
    ds = str(desired_state)
    for i, s in enumerate(states):
        if str(s) == ds:
            return i

    # fallback: if binary and not found, try index 1
    if len(states) == 2:
        return 1

    raise ValueError(f"State {desired_state} not found in {var_name}. Available states: {states}")

# ---------------------------------------------------------------

"""
    Calculates P(Overheat = positive_state | evidence).

    - infer: VariableElimination(model) object.
    - evidence: dict with sensors (e.g., {"SpindleTemp":2, "Vibration":1, "CoolantFlow":0}).
    - positive_state: usually 1 (for Overheat 0/1). If "yes", use "yes".
"""
def overheat_probability(infer, evidence, target, positive_state=1):
    q = infer.query(variables=[target], evidence=evidence, show_progress=False)
    idx = state_index(q, target, positive_state)
    return float(q.values[idx])

# ---------------------------------------------------------------

"""
    Makes a predition for ONE row of the dataframe:

    1) Creates evidence with the sensor values of that row
    2) Calculates p = P(Overheat=1 | evidence)
    3) Converts to class 0/1 using threshold
"""
def predict_overheat_row(row, infer, features, target, threshold=0.5, positive_state=1):
    evidence = {f: row[f] for f in features}
    p1 = overheat_probability(infer, evidence, target=target, positive_state=positive_state)
    yhat = 1 if p1 >= threshold else 0
    return yhat, p1

# ---------------------------------------------------------------

"""
    Makes a prediction for a whole dataframe.
    Returns:
      - y_pred: list of classes (0/1)
      - y_prob: list of probabilities P(Overheat=1)
"""
def predict_overheat_df(df, model, features, target, threshold=0.5, positive_state=1):
    infer = VariableElimination(model)

    y_pred, y_prob = [], []
    for _, row in df.iterrows():
        pred, prob = predict_overheat_row(
            row, infer,
            features=features, target=target,
            threshold=threshold, positive_state=positive_state
        )
        y_pred.append(pred)
        y_prob.append(prob)

    return np.array(y_pred), np.array(y_prob)

# ---------------------------------------------------------------

"""
    Calculates metrics and (optionally) prints them.

    - y_true: array with true labels (0/1)
    - y_pred: array with predicted classes (0/1)
    - y_prob: array with probabilities (for AUC). Can be None.
"""
def evaluate_predictions(y_true, y_pred, y_prob=None, verbose=True):
    cm = confusion_matrix(y_true, y_pred)

    out = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ConfusionMatrix": cm.tolist(),
        "AUC": None
    }

    # AUC only makes sense if you have probabilities and both classes exist in y_true
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["AUC"] = float(roc_auc_score(y_true, y_prob))

    if verbose:
        print("Confusion matrix [[TN, FP],[FN, TP]]:")
        print(cm)
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=3))
        print("Metrics summary:", {k: v for k, v in out.items() if k != "ConfusionMatrix"})
    return out

# ---------------------------------------------------------------

"""
    Created the BN with latent causes -> sensors -> Overheat.
"""
def build_em_model(latent_causes):
    edges = [
        ("BearingWear", "Vibration"),
        ("FanFault", "SpindleTemp"),
        ("CoolingEfficiency", "SpindleTemp"),
        ("CloggedFilter", "CoolantFlow"),

        ("Vibration", "Overheat"),
        ("SpindleTemp", "Overheat"),
        ("CoolantFlow", "Overheat"),
    ]

    # latents=set(latent_causes) serves to document what are latent variables
    return DiscreteBayesianNetwork(edges, latents=set(latent_causes))

# ---------------------------------------------------------------

"""
    Learns the CPDs of the model using Expectation-Maximization (EM).
    EM is necessary beacuse the latent causes do not exist in the dataframe.
    The algorithm alternates:
      - E-step: infers the distribution of the latents given the observed
      - M-step: updates CPDs maximizing the expected likelihood

    Parameters:
      - latent_card: cardinality (number of states) of each latent.
        Binary example: {latent: 2} -> (0=ok, 1=fault)
      - max_iter: EM iterations
      - seed: for reproducible results
"""
def fit_em(model, train_df, latents, features, target, latent_card=None, max_iter=100, seed=42, atol=1e-12, show_progress=True):

    display(Markdown("**[fit_em] START**"))

    cols = features + [target]
    data = train_df[cols].dropna().copy()

    if latent_card is None:
        latent_card = {lv: 2 for lv in latents}

    em = ExpectationMaximization(model, data)
    cpds = em.get_parameters(latent_card=latent_card, max_iter=max_iter, seed=seed, atol=atol, show_progress=show_progress)


    model.add_cpds(*cpds)
    model.check_model()

    display(Markdown(f"**[fit_em] FINISHED** — CPDs: {len(model.get_cpds())}"))
    return model


# ---------------------------------------------------------------

"""
    Diagnoses a single case given the evidence from sensors.

    Input:
      - evidence: dict with sensor values, ex:
          {"SpindleTemp": 2, "Vibration": 1, "CoolantFlow": 0}
      - threshold: only ranks causes if P(Overheat=1|evidence) >= threshold

    Output (dict):
      {
        "evidence": evidence,
        "p_overheat": <float>,
        "do_causes": <bool>,
        "causes_ranking": [(latent, p_fault), ...]  ou None
      }

    Note:
      - If overheat is not probable, it does not calculate ranking (causes_ranking=None).
      - Ranking is calculated as P(latent=fault | evidence, Overheat=1).
"""
def diagnose_case(model, evidence, target, latents, threshold=0.5, overheat_state=1, latent_fault_state=1):
    infer = VariableElimination(model)

    # Probability of overheat
    p_oh = overheat_probability(infer, evidence, target, positive_state=overheat_state)

    # Decides if it is worth doing root-cause analysis
    do_causes = (p_oh >= threshold)
    ranking = None

    if do_causes:
        
        # Conditions on Overheat=1 for "cause of overheat"
        ev = dict(evidence)
        ev[target] = overheat_state

        ranking = []
        for lv in latents:
            q = infer.query(variables=[lv], evidence=ev, show_progress=False)
            idx = state_index(q, lv, latent_fault_state)
            p_fault = float(q.values[idx])
            ranking.append((lv, p_fault))

        ranking.sort(key=lambda x: x[1], reverse=True)

    return {
        "evidence": dict(evidence),
        "p_overheat": float(p_oh),
        "do_causes": bool(do_causes),
        "causes_ranking": ranking,
        "threshold": float(threshold),
        "overheat_state": overheat_state,
        "latent_fault_state": latent_fault_state,
    }

# ---------------------------------------------------------------
"""
    Faz print do resultado do diagnose_case().

    - result: dict devolvido por diagnose_case
    - top_k: quantas causas mostrar (se houver ranking)
"""

def print_diagnosis(model, evidence, target, latents, threshold=0.5, overheat_state=1, latent_fault_state=1, top_k=4):
    result = diagnose_case(model, evidence, target, latents, threshold, overheat_state, latent_fault_state)
    
    evidence = result["evidence"]
    p_oh = result["p_overheat"]
    do_causes = result["do_causes"]
    ranking = result["causes_ranking"]
    threshold = result["threshold"]
    overheat_state = result["overheat_state"]

    print("Evidence (sensors):", evidence)
    print(f"P(Overheat={overheat_state} | evidence) = {p_oh:.4f}")

    if not do_causes:
        print(f"Without cause ranking (p<{threshold}).")
        return

    print("\nRanking of latent causes (conditioned on Overheat=1):")
    for lv, p in ranking[:top_k]:
        print(f"  - {lv}: P(fault) = {p:.4f}")

# ---------------------------------------------------------------

"""
    Chooses as 'faulty' the latent state that maximizes P(sensor=bad | latent=state).
"""
def infer_fault_state_by_sensor(model, latent, sensor, bad_sensor_state):
    infer = VariableElimination(model)

    scores = {}
    for s in [0, 1]:
        q = infer.query([sensor], evidence={latent: s}, show_progress=False)
        idx = state_index(q, sensor, bad_sensor_state)
        scores[s] = float(q.values[idx])

    # escolhe o maior
    fault_state = max(scores, key=scores.get)

    print(f"[{latent}] bad={sensor}=={bad_sensor_state}")
    print(f"  P(bad | {latent}=0) = {scores[0]:.4f}")
    print(f"  P(bad | {latent}=1) = {scores[1]:.4f}")
    print(f"  => fault_state = {fault_state} (difference = {abs(scores[1]-scores[0]):.4f})")

    return fault_state

# ---------------------------------------------------------------

def anchor_table_two_parents(model, y, y_bad, a, b):
    infer = VariableElimination(model)
    tbl = {}
    for sa in [0,1]:
        for sb in [0,1]:
            q = infer.query([y], evidence={a: sa, b: sb}, show_progress=False)
            p = float(q.values[state_index(q, y, y_bad)])
            tbl[(sa, sb)] = p

    print(f"\nP({y}={y_bad} | {a}, {b})")
    print(f"          {b}=0      {b}=1")
    print(f"{a}=0   {tbl[(0,0)]:.4f}    {tbl[(0,1)]:.4f}")
    print(f"{a}=1   {tbl[(1,0)]:.4f}    {tbl[(1,1)]:.4f}")
    return tbl

# --------------------------------------------------------------

"""
    Switches the states of a binary variable in the model.
    This forces the interpretation to invert (what was state 0 becomes state 1).
"""
def flip_binary_state_in_model(model, var):
    new_cpds = []

    for cpd in model.get_cpds():
        # If the variable does not appear in this CPD, keep it.
        if var not in cpd.variables:
            new_cpds.append(cpd)
            continue

        # Convert CPD to multidimensional factor for safe flipping along axis
        factor = cpd.to_factor()

        # axis corresponding to 'var' within the factor
        axis = factor.variables.index(var)

        # Flip along this axis (swap state 0 with state 1)
        factor.values = np.flip(factor.values, axis=axis)

        # Rebuild TabularCPD with the same "target" variable and same parents
        evidence = getattr(cpd, "evidence", None)
        if evidence is None:
            evidence = list(cpd.variables[1:])  # fallback

        evidence_card = getattr(cpd, "evidence_card", None)
        if evidence_card is None:
            evidence_card = list(cpd.cardinality[1:])  # fallback

        # Factor -> 2D values in the format expected by TabularCPD
        new_values = factor.values.reshape(int(cpd.variable_card), -1)

        new_cpd = TabularCPD(
            variable=cpd.variable,
            variable_card=int(cpd.variable_card),
            values=new_values,
            evidence=evidence if evidence else None,
            evidence_card=[int(x) for x in evidence_card] if evidence_card else None,
            state_names=getattr(cpd, "state_names", None),
        )
        new_cpds.append(new_cpd)

    # Replace all CPDs in the model
    model.remove_cpds(*model.get_cpds())
    model.add_cpds(*new_cpds)
    model.check_model()
    return model

# ---------------------------------------------------------------

def fit_em_with_init(model, train_df, latents, features, target, init_cpds=None, latent_card=None, max_iter=100, seed=42, atol=1e-12, show_progress=True):
    data = train_df[features + [target]].dropna().copy()

    if latent_card is None:
        latent_card = {lv: 2 for lv in latents}

    em = ExpectationMaximization(model, data)

    cpds = em.get_parameters(
        latent_card=latent_card,
        max_iter=max_iter,
        seed=seed,
        atol=atol,
        init_cpds=init_cpds,   # <-- dict here
        show_progress=show_progress
    )

    model.add_cpds(*cpds)
    model.check_model()
    return model
