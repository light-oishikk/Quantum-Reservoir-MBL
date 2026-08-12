# Quantum Reservoir Computing for NH-MBL

This repository contains the fully parallelized, rigorously tested experiment suite for identifying Disorder-Driven Spectral Transitions in Non-Hermitian Many-Body Systems using Quantum Reservoir Computing (QRC).

## Files Overview
* `nh_mbl_core.py`: The core physics engine. Builds the Non-Hermitian Hamiltonians, handles exact diagonalization (ED), and parallelizes the quantum dynamics simulations using `joblib`.
* `run_experiments.py`: The machine learning orchestration script. Runs all the benchmarking experiments, evaluates models (including the fixed fair-comparison Classical ESN baseline), and outputs results.
* `results/`: Contains the CSVs and plot outputs for the experiments.

## Why Python Scripts instead of Jupyter Notebooks?
Due to the heavy memory constraints of exact diagonalization for complex many-body systems (e.g., $N=10$, $1024 \times 1024$ dense complex matrices), running parallel processes inside a Jupyter Notebook causes frequent crashes and memory leaks. Converting to modular `.py` files is standard for high-performance physics simulations and ensures absolute stability while generating datasets with 1000s of realizations.

## Running the Code
Ensure you have the required dependencies:
```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib
```
Then execute the main runner:
```bash
python run_experiments.py
```
