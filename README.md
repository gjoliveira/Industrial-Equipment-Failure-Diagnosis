# Hybrid Neuro-Symbolic System for Industrial Equipment Failure Diagnosis

This project implements a decision-support system for diagnosing failures in a CNC milling machine's spindle system. It combines **Bayesian Networks (BN)** for probabilistic reasoning under uncertainty with **Knowledge Graphs (KG)** for domain knowledge representation and explanation.

## Project Structure

-   `notebook.ipynb`: The main Jupyter Notebook containing the complete implementation of the pipeline.
-   `data/`: Directory containing the dataset files (telemetry, labels, KG master data).
-   `app.py`: A Streamlit web application for interactive diagnostics.
-   `knowledge_base.ttl`: The generated Knowledge Graph in Turtle format.

## Components Description

The system consists of three main stages, implemented sequentially in the notebook:

### 1. Data Processing & Knowledge Graph Construction
-   **Input:** CSV files defining components, failure modes, symptoms, and their relationships (`components.csv`, `causes.csv`, `symptoms.csv`, `relations.csv`).
-   **Process:**
    -   Uses `rdflib` to build an RDF graph.
    -   Defines an ontology with classes (`Component`, `FailureMode`, `Symptom`) and properties (`manifestsAs`, `affectsComponent`).
    -   Ingests maintenance procedures (`procedures.csv`) with associated costs and efforts.
-   **Output:** A queryable Knowledge Graph capable of reasoning about machine structure and repair actions.

### 2. Probabilistic Reasoning (Bayesian Network)
-   **Input:** Discrete telemetry data (`telemetry_discrete.csv`) representing sensor readings (e.g., 'High' Vibration, 'Low' CoolantFlow).
-   **Library:** `pgmpy`
-   **Structure:** A causal network where Latent Causes (e.g., `BearingWearHigh`) influence Observed Symptoms (e.g., `vibration_rms`).
-   **Learning:**
    -   Due to the latent nature of failure causes in the provided dataset, the network parameters (CPDs) are initialized with expert knowledge (randomized valid distributions for the prototype) to model the conditional probabilities $P(Symptom | Cause)$.
-   **Inference:** Uses Variable Elimination to compute the posterior probability of each failure mode given current sensor evidence.

### 3. Decision Support & Integration
-   **Integration:** The system links the probabilistic output of the BN with the semantic knowledge in the KG.
-   **Logic:**
    1.  **Diagnosis:** The BN estimates the probability of each failure mode.
    2.  **Prescription:** The system queries the KG to find maintenance actions that *mitigate* the likely failure modes.
    3.  **Recommendation:** It calculates a score based on the **Risk** (Probability $\times$ Impact) and the **Cost** of the action (Financial Cost + Time Effort) to recommend the most optimal maintenance procedure.

## How to Run

### Prerequisites
Install the required Python libraries:
```bash
pip install pandas numpy matplotlib seaborn networkx rdflib pgmpy streamlit
```

### Running the Notebook
Open `notebook.ipynb` in Jupyter and run all cells to execute the full pipeline, generate the Knowledge Graph, and see example diagnostics.

### Running the Streamlit App
To launch the interactive dashboard:
```bash
streamlit run app.py
```
This app allows you to simulate sensor readings and view real-time diagnostic probabilities and maintenance recommendations.
