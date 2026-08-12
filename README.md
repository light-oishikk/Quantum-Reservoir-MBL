# Quantum Reservoir Computing for NH-MBL

This repository contains the parallelized experiment suite for identifying Disorder-Driven Spectral Transitions in Non-Hermitian Many-Body Systems using Quantum Reservoir Computing (QRC).

## Files Overview
* `nh_mbl_core.py`: The core physics engine (builds Hamiltonians and runs quantum dynamics).
* `run_experiments.py`: The machine learning orchestration script (runs all benchmarking experiments and outputs results).
* `results/`: Contains the CSVs and plot outputs for the experiments.

## Running the Code

1. Install dependencies:
```bash
pip install numpy scipy scikit-learn pandas matplotlib joblib
```

2. Execute the main runner:
```bash
python run_experiments.py
```
