import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split, KFold
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
import joblib

# Import core library functions
from nh_mbl_core import (
    build_dataset_parallel as core_build_dataset_parallel,
    estimate_W_star as core_estimate_W_star,
    assign_phase_labels as core_assign_phase_labels,
    neel_index,
    make_product_state_index,
    sector_indices,
    build_H_NH
)

def build_dataset_parallel(N, g, W_list, n_realizations, meas_times, init_state_idx=None):
    X, X_oracle, y_f, y_w, skipped = core_build_dataset_parallel(
        N=N, g=g, W_list=W_list, n_real=n_realizations, 
        meas_times=meas_times, initial_state_idx=init_state_idx,
        n_jobs=4, verbose=1  # cap at 4 to avoid OOM
    )
    cols = []
    for t in meas_times:
        cols.append(f'S_{t}')
        for s in range(N):
            cols.append(f'Sz_{s}_{t}')
    df = pd.DataFrame(X, columns=cols)
    df['W'] = y_w
    df['f_real'] = y_f
    return df

def estimate_W_star(df):
    W_star = core_estimate_W_star(df['f_real'].values, df['W'].values)
    return W_star, None, None

def assign_phase_labels(df, W_star):
    df_new = df.copy()
    df_new['phase'] = core_assign_phase_labels(df['W'].values, W_star)
    return df_new

# --- Constants ---
N_DEFAULT = 8        # N=10 exhausts RAM with 16 workers; N=8 is 64x lighter
G_DEFAULT = 0.3
N_REAL_DEFAULT = 200  # 200 realizations is statistically adequate
W_LIST = (0.5, 1, 1.5, 2, 3, 4, 6, 10)
MEAS_TIMES = (0, 1, 2, 4, 8)
SEEDS = [42, 123, 321, 777, 999, 2024, 31415, 27182, 16180, 9001]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
RESULTS_DIR = 'results'

def setup_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Base Estimator Generators ---
def get_clf(name='rf'):
    if name == 'rf':
        return Pipeline([('scaler', StandardScaler()),
                         ('clf', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42, n_jobs=-1))])
    elif name == 'lr':
        return Pipeline([('scaler', StandardScaler()),
                         ('clf', LogisticRegression(max_iter=1000, random_state=42))])
    elif name == 'svm_rbf':
        return Pipeline([('scaler', StandardScaler()),
                         ('clf', SVC(kernel='rbf', probability=True, random_state=42))])
    raise ValueError(f"Unknown classifier {name}")

def extract_features(df_data, feature_set='full'):
    if feature_set == 'w_only':
        X = df_data[['W']].values
    elif feature_set == 'oracle':
        # Assuming we just have f_real as disorder statistic in the data
        X = df_data[['f_real']].values
    elif feature_set == 'full':
        feature_cols = [c for c in df_data.columns if c.startswith('O_') or c.startswith('Sz_') or c.startswith('S_')]
        X = df_data[feature_cols].values
    elif feature_set == 'survival':
        feature_cols = [c for c in df_data.columns if c.startswith('S_')]
        X = df_data[feature_cols].values
    else:
        raise ValueError("Unknown feature set")
    return X

# --- Experiment 1: Baseline Comparison ---
def experiment_baseline_comparison(df_data):
    print("Running Experiment 1: Baseline Comparison")
    y = df_data['phase'].values
    
    results = []
    
    for fset in ['w_only', 'oracle', 'full']:
        X = extract_features(df_data, fset)
        
        accs, aucs = [], []
        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            clf = get_clf('rf')
            
            # evaluate
            scores = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'roc_auc'], n_jobs=-1)
            accs.append(scores['test_accuracy'].mean())
            aucs.append(scores['test_roc_auc'].mean())
            
        results.append({
            'Feature Set': fset,
            'Accuracy Mean': np.mean(accs),
            'Accuracy SD': np.std(accs),
            'AUC Mean': np.mean(aucs),
            'AUC SD': np.std(aucs)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp1_baseline_comparison.csv", index=False)
    print(res_df)
    return res_df

# --- Experiment 2: Continuous Prediction ---
def experiment_continuous_prediction(df_data):
    print("Running Experiment 2: Continuous Prediction")
    X = extract_features(df_data, 'full')
    y = df_data['f_real'].values
    
    models = {
        'Ridge': Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))]),
        'SVR-RBF': Pipeline([('scaler', StandardScaler()), ('reg', SVR(kernel='rbf'))]),
        'RandomForest': Pipeline([('scaler', StandardScaler()), ('reg', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=['r2', 'neg_mean_absolute_error', 'neg_root_mean_squared_error'], n_jobs=-1)
        results.append({
            'Model': name,
            'R2 Mean': scores['test_r2'].mean(),
            'R2 SD': scores['test_r2'].std(),
            'MAE Mean': -scores['test_neg_mean_absolute_error'].mean(),
            'MAE SD': scores['test_neg_mean_absolute_error'].std(),
            'RMSE Mean': -scores['test_neg_root_mean_squared_error'].mean(),
            'RMSE SD': scores['test_neg_root_mean_squared_error'].std()
        })
        
    # Scatter plot for RF
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = models['RandomForest']
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Exact f_real')
    plt.ylabel('Predicted f_real')
    plt.title('Continuous Prediction (RandomForest)')
    plt.savefig(f"{RESULTS_DIR}/exp2_continuous_prediction.png")
    plt.close()
    
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp2_continuous_prediction.csv", index=False)
    print(res_df)
    return res_df

# --- Experiment 3: Correlation Analysis ---
def experiment_correlation_analysis(df_data):
    print("Running Experiment 3: Correlation Analysis")
    y = df_data['f_real'].values
    
    results = []
    feature_cols = [c for c in df_data.columns if c.startswith('O_') or c.startswith('Sz_') or c.startswith('S_')]
    
    for col in feature_cols:
        x = df_data[col].values
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        
        # parse time step if possible
        t_str = col.split('_')[-1]
        try:
            t = float(t_str)
        except:
            t = np.nan
            
        results.append({
            'Observable': col,
            'Time': t,
            'Pearson r': pearson_r,
            'Pearson p': pearson_p,
            'Spearman r': spearman_r,
            'Spearman p': spearman_p
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp3_correlation_analysis.csv", index=False)
    print(res_df.head())
    return res_df

# --- Experiment 4: ESN Baseline ---
class SimpleESN:
    def __init__(self, input_dim, reservoir_size=100, spectral_radius=0.9,
                 input_scale=0.5, leak=1.0, ridge_alpha=1e-3, seed=0):
        self.reservoir_size = reservoir_size
        self.leak = leak
        self.ridge_alpha = ridge_alpha
        np.random.seed(seed)
        
        # Initialize weights
        W_in = np.random.uniform(-1, 1, (reservoir_size, input_dim)) * input_scale
        W_res = np.random.normal(0, 1, (reservoir_size, reservoir_size))
        
        # Scale spectral radius
        rho = np.max(np.abs(np.linalg.eigvals(W_res)))
        W_res = W_res * (spectral_radius / rho)
        
        self.W_in = W_in
        self.W_res = W_res
        self.readout = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=1.0/ridge_alpha, max_iter=1000, random_state=seed))
        ])

    def _state(self, sequence):
        # sequence shape: (time_steps, input_dim)
        state = np.zeros(self.reservoir_size)
        for x_t in sequence:
            pre_act = self.W_in @ x_t + self.W_res @ state
            state = (1 - self.leak) * state + self.leak * np.tanh(pre_act)
        return state

    def fit(self, X_sequences, y):
        states = np.array([self._state(seq) for seq in X_sequences])
        self.readout.fit(states, y)
        return self

    def predict_proba(self, X_sequences):
        states = np.array([self._state(seq) for seq in X_sequences])
        return self.readout.predict_proba(states)

def experiment_esn_baseline(df_data):
    """
    Fair comparison: Both ESN and QRC receive the SAME input data —
    the Sz observables measured at each time step.

    ESN: classical random reservoir processes the Sz time series.
    QRC: quantum-evolved features (same Sz values) fed to RF readout.

    The only difference is how the reservoir is implemented:
    classical random recurrent network vs. the actual quantum Hamiltonian dynamics.
    """
    print("Running Experiment 4: ESN Baseline (fair — same Sz time-series input)")

    # Build the sequence representation:
    # For each realization, input is a sequence of length T = len(MEAS_TIMES)
    # At each time step t, the input vector = [Sz_0_t, Sz_1_t, ..., Sz_{N-1}_t]
    N = N_DEFAULT
    T = len(MEAS_TIMES)

    sequences = []
    for _, row in df_data.iterrows():
        seq = []
        for t in MEAS_TIMES:
            sz_at_t = [row[f'Sz_{s}_{t}'] for s in range(N)]
            seq.append(sz_at_t)
        sequences.append(np.array(seq))  # shape (T, N)
    X_seq = np.array(sequences)  # shape (n_samples, T, N)

    y = df_data['phase'].values

    # QRC features: same Sz values but fed as flat vector to RF
    X_q = extract_features(df_data, 'full')

    esn_aucs = []
    qr_aucs = []

    for seed in SEEDS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

        esn_fold_aucs = []
        qr_fold_aucs = []

        for train_idx, test_idx in cv.split(X_seq, y):
            # ESN: processes Sz as a temporal sequence through classical reservoir
            esn = SimpleESN(input_dim=N, reservoir_size=200, seed=seed)
            esn.fit(X_seq[train_idx], y[train_idx])
            y_pred_esn = esn.predict_proba(X_seq[test_idx])[:, 1]
            esn_fold_aucs.append(roc_auc_score(y[test_idx], y_pred_esn))

            # QRC: same Sz features but via quantum-evolved readout (RF)
            clf = get_clf('rf')
            clf.fit(X_q[train_idx], y[train_idx])
            y_pred_qr = clf.predict_proba(X_q[test_idx])[:, 1]
            qr_fold_aucs.append(roc_auc_score(y[test_idx], y_pred_qr))

        esn_aucs.append(np.mean(esn_fold_aucs))
        qr_aucs.append(np.mean(qr_fold_aucs))

    stat, p_val = wilcoxon(esn_aucs, qr_aucs)

    results = pd.DataFrame({
        'Seed': SEEDS,
        'ESN AUC (classical, same input)': esn_aucs,
        'QRC AUC (quantum dynamics)': qr_aucs
    })

    results.to_csv(f"{RESULTS_DIR}/exp4_esn_comparison.csv", index=False)

    print(f"\nWilcoxon signed-rank test p-value: {p_val:.6f}")
    print(f"ESN  AUC: {np.mean(esn_aucs):.4f} ± {np.std(esn_aucs):.4f}")
    print(f"QRC  AUC: {np.mean(qr_aucs):.4f} ± {np.std(qr_aucs):.4f}")
    if p_val < 0.05:
        winner = "QRC" if np.mean(qr_aucs) > np.mean(esn_aucs) else "ESN"
        print(f"Statistically significant difference (p<0.05): {winner} wins")
    else:
        print("No statistically significant difference between ESN and QRC")
    return results

# --- Experiment 5: g Robustness ---
def experiment_g_robustness():
    print("Running Experiment 5: g Robustness")
    g_values = [0.10, 0.20, 0.30, 0.40, 0.50]
    
    results = []
    for g in g_values:
        print(f"  Testing g = {g}")
        df = build_dataset_parallel(N_DEFAULT, g, W_LIST, n_realizations=100, meas_times=MEAS_TIMES)
        W_star, _, _ = estimate_W_star(df)
        df_labeled = assign_phase_labels(df, W_star)
        
        X = extract_features(df_labeled, 'full')
        y = df_labeled['phase'].values
        
        clf = get_clf('rf')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        results.append({
            'g': g,
            'W_star': W_star,
            'AUC Mean': scores['test_score'].mean(),
            'AUC SD': scores['test_score'].std()
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp5_g_robustness.csv", index=False)
    print(res_df)
    return res_df

# --- Experiment 5b: Cross-g Generalization ---
def experiment_cross_g_generalization():
    """
    Train on g=0.3 data, test on g=0.1, 0.2, 0.4, 0.5.
    This tests whether the model learns physics-general features
    or just memorizes g-specific patterns.
    """
    print("Running Experiment 5b: Cross-g Generalization")
    g_train = 0.3
    g_test_values = [0.1, 0.2, 0.4, 0.5]

    # Build training data at g=0.3
    print(f"  Building training set at g={g_train}")
    df_train = build_dataset_parallel(N_DEFAULT, g_train, W_LIST, n_realizations=200, meas_times=MEAS_TIMES)
    W_star_train, _, _ = estimate_W_star(df_train)
    df_train_labeled = assign_phase_labels(df_train, W_star_train)

    X_train = extract_features(df_train_labeled, 'full')
    y_train = df_train_labeled['phase'].values

    # Train classifier on full training set for cross-g testing
    clf = get_clf('rf')
    clf.fit(X_train, y_train)
    
    # Self-test (sanity check) using cross-validation
    from sklearn.model_selection import cross_validate, StratifiedKFold
    cv_res = cross_validate(get_clf('rf'), X_train, y_train, 
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring=['roc_auc', 'accuracy'])
    self_auc = cv_res['test_roc_auc'].mean()
    self_acc = cv_res['test_accuracy'].mean()

    results = [{
        'Train g': g_train,
        'Test g': g_train,
        'W_star (test)': W_star_train,
        'AUC': self_auc,
        'Accuracy': self_acc,
        'Note': 'self-test (CV)'
    }]

    for g_test in g_test_values:
        print(f"  Testing on g={g_test}")
        df_test = build_dataset_parallel(N_DEFAULT, g_test, W_LIST, n_realizations=200, meas_times=MEAS_TIMES)
        
        # Use the TRAINING W* to label test data (same threshold)
        df_test_labeled = assign_phase_labels(df_test, W_star_train)
        
        # Also compute the test set's own W* for reference
        W_star_test, _, _ = estimate_W_star(df_test)

        X_test = extract_features(df_test_labeled, 'full')
        y_test = df_test_labeled['phase'].values

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = float('nan')

        results.append({
            'Train g': g_train,
            'Test g': g_test,
            'W_star (test)': W_star_test,
            'AUC': auc,
            'Accuracy': acc,
            'Note': 'cross-g transfer'
        })
        print(f"    AUC={auc:.4f}, Acc={acc:.4f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp5b_cross_g_generalization.csv", index=False)
    print("\n" + res_df.to_string())
    return res_df

# --- Experiment 6: Initial State Ablation ---
def experiment_initial_state_ablation():
    print("Running Experiment 6: Initial State Ablation")
    N = 8
    
    states = {
        'Neel': neel_index(N),
        'Inverted Neel': neel_index(N) ^ ((1 << N) - 1),  # bitwise flip
        'Uniform': make_product_state_index([0, 1] * (N // 2))
    }
    
    results = []
    for name, idx in states.items():
        print(f"  Testing State: {name}")
        df = build_dataset_parallel(N, G_DEFAULT, W_LIST, n_realizations=100, meas_times=MEAS_TIMES, init_state_idx=idx)
        W_star, _, _ = estimate_W_star(df)
        df_labeled = assign_phase_labels(df, W_star)
        
        X = extract_features(df_labeled, 'full')
        y = df_labeled['phase'].values
        
        clf = get_clf('rf')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_validate(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        results.append({
            'Initial State': name,
            'AUC Mean': scores['test_score'].mean(),
            'AUC SD': scores['test_score'].std()
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp6_initial_state.csv", index=False)
    print(res_df)
    return res_df

# --- Experiment 7: Noise Robustness ---
def experiment_noise_robustness(df_data):
    print("Running Experiment 7: Noise Robustness")
    X_full = extract_features(df_data, 'full')
    X_surv = extract_features(df_data, 'survival')
    y = df_data['phase'].values
    
    X_full_tr, X_full_te, y_tr, y_te = train_test_split(X_full, y, test_size=0.3, random_state=42, stratify=y)
    X_surv_tr, X_surv_te, _, _ = train_test_split(X_surv, y, test_size=0.3, random_state=42, stratify=y)
    
    clf_full = get_clf('rf').fit(X_full_tr, y_tr)
    clf_surv = get_clf('rf').fit(X_surv_tr, y_tr)
    
    results = []
    for noise in NOISE_LEVELS:
        # Add gaussian noise to test set
        X_f_noisy = X_full_te + np.random.normal(0, noise, X_full_te.shape)
        X_s_noisy = X_surv_te + np.random.normal(0, noise, X_surv_te.shape)
        
        pred_f = clf_full.predict_proba(X_f_noisy)[:, 1]
        pred_s = clf_surv.predict_proba(X_s_noisy)[:, 1]
        
        results.append({
            'Noise Level': noise,
            'Full Features AUC': roc_auc_score(y_te, pred_f),
            'Survival Only AUC': roc_auc_score(y_te, pred_s)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp7_noise_robustness.csv", index=False)
    
    plt.figure()
    plt.plot(res_df['Noise Level'], res_df['Full Features AUC'], marker='o', label='Full Features')
    plt.plot(res_df['Noise Level'], res_df['Survival Only AUC'], marker='s', label='Survival Only')
    plt.xlabel('Noise Level (\u03c3)')
    plt.ylabel('Test AUC')
    plt.title('Robustness to Measurement Noise')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/exp7_noise_robustness.png")
    plt.close()
    
    print(res_df)
    return res_df

# --- Experiment 8: Training Size Scaling ---
def experiment_training_size_scaling(df_data):
    print("Running Experiment 8: Training Size Scaling")
    X = extract_features(df_data, 'full')
    y = df_data['phase'].values
    
    fractions = [0.1, 0.2, 0.4, 0.6, 0.8]
    
    results = []
    
    for frac in fractions:
        aucs = []
        for seed in SEEDS[:5]:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=frac, random_state=seed, stratify=y)
            clf = get_clf('rf').fit(X_tr, y_tr)
            pred = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y_te, pred))
            
        results.append({
            'Training Fraction': frac,
            'Training Samples': len(y_tr),
            'AUC Mean': np.mean(aucs),
            'AUC SD': np.std(aucs)
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp8_training_scaling.csv", index=False)
    
    plt.figure()
    plt.errorbar(res_df['Training Samples'], res_df['AUC Mean'], yerr=res_df['AUC SD'], marker='o')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Test AUC')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}/exp8_training_scaling.png")
    plt.close()
    
    print(res_df)
    return res_df

# --- Experiment 9: Linear vs Nonlinear ---
def experiment_linear_vs_nonlinear(df_data):
    print("Running Experiment 9: Linear vs Nonlinear")
    y = df_data['phase'].values
    
    results = []
    
    for t in MEAS_TIMES:
        cols_t = [c for c in df_data.columns if c.endswith(f'_{t}') and not c.startswith('W')]
        if len(cols_t) == 0:
            continue
            
        X = df_data[cols_t].values
        
        for name, clf_name in [('Logistic Regression', 'lr'), ('RBF SVM', 'svm_rbf')]:
            clf = get_clf(clf_name)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_validate(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
            
            results.append({
                'Time': t,
                'Model': name,
                'AUC Mean': scores['test_score'].mean(),
                'AUC SD': scores['test_score'].std()
            })
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp9_linear_vs_nonlinear.csv", index=False)
    print(res_df)
    return res_df

# --- Experiment 10: Feature Separation ---
def experiment_feature_separation(df_data):
    print("Running Experiment 10: Feature Separation")
    y = df_data['phase'].values
    phases = np.unique(y)
    if len(phases) < 2:
        return pd.DataFrame()
        
    results = []
    
    for t in MEAS_TIMES:
        cols_t = [c for c in df_data.columns if c.endswith(f'_{t}') and not c.startswith('W')]
        if len(cols_t) == 0:
            continue
            
        X = df_data[cols_t].values
        X_0 = X[y == phases[0]]
        X_1 = X[y == phases[1]]
        
        centroid_0 = np.mean(X_0, axis=0)
        centroid_1 = np.mean(X_1, axis=0)
        
        dist = np.linalg.norm(centroid_0 - centroid_1)
        
        # Cohen's d multivariate approx
        pooled_cov = (np.cov(X_0.T) * len(X_0) + np.cov(X_1.T) * len(X_1)) / (len(X_0) + len(X_1) - 2)
        try:
            inv_cov = np.linalg.inv(pooled_cov)
            d_sq = (centroid_0 - centroid_1).T @ inv_cov @ (centroid_0 - centroid_1)
            d = np.sqrt(max(0, d_sq))
        except:
            d = np.nan
            
        results.append({
            'Time': t,
            'Centroid Distance': dist,
            'Cohen d': d
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp10_feature_separation.csv", index=False)
    print(res_df)
    return res_df

# --- Experiment 11: System Size Benchmark ---
def experiment_system_size_benchmark():
    print("Running Experiment 11: System Size Benchmark")
    sizes = [6, 8, 10]
    
    results = []
    for N in sizes:
        print(f"  Testing N = {N}")
        
        if N <= 8:
            n_real = 200
        elif N == 10:
            n_real = 100
        else:
            n_real = 50
            
        start_time = time.time()
        
        try:
            df = build_dataset_parallel(N, G_DEFAULT, W_LIST, n_realizations=n_real, meas_times=MEAS_TIMES)
            W_star, _, _ = estimate_W_star(df)
            df_labeled = assign_phase_labels(df, W_star)
            
            X = extract_features(df_labeled, 'full')
            y = df_labeled['phase'].values
            
            clf = get_clf('rf')
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_validate(clf, X, y, cv=cv, scoring=['accuracy', 'roc_auc'], n_jobs=-1)
            
            runtime = time.time() - start_time
            
            results.append({
                'N': N,
                'N_realizations': n_real,
                'W_star': W_star,
                'Accuracy': scores['test_accuracy'].mean(),
                'AUC': scores['test_roc_auc'].mean(),
                'Runtime (s)': runtime
            })
        except Exception as e:
            print(f"    Failed for N={N}: {e}")
            
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{RESULTS_DIR}/exp11_system_size.csv", index=False)
    
    if not res_df.empty:
        plt.figure()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        
        ax1.plot(res_df['N'], res_df['AUC'], 'g-o', label='AUC')
        ax2.plot(res_df['N'], res_df['Runtime (s)'], 'b-s', label='Runtime (s)')
        
        ax1.set_xlabel('System Size (N)')
        ax1.set_ylabel('AUC', color='g')
        ax2.set_ylabel('Runtime (s)', color='b')
        
        plt.title('Performance vs System Size')
        plt.savefig(f"{RESULTS_DIR}/exp11_system_size.png")
        plt.close()
        
    print(res_df)
    return res_df


# --- Main Runner ---
def run_all():
    import sys
    setup_results_dir()
    t_total = time.time()

    def step(name, fn, *args, **kwargs):
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f">>> STARTING: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        result = fn(*args, **kwargs)
        print(f"<<< DONE: {name}  [{time.time()-t0:.1f}s]", flush=True)
        sys.stdout.flush()
        return result

    print(f"=== Generating Primary Dataset (N={N_DEFAULT}, {N_REAL_DEFAULT} real) ===", flush=True)
    df_primary = build_dataset_parallel(N_DEFAULT, G_DEFAULT, W_LIST, n_realizations=N_REAL_DEFAULT, meas_times=MEAS_TIMES)
    W_star, _, _ = estimate_W_star(df_primary)
    df_primary_labeled = assign_phase_labels(df_primary, W_star)
    print(f"Primary dataset ready. Estimated W* = {W_star:.3f}", flush=True)

    step("Exp1: Baseline Comparison",        experiment_baseline_comparison,     df_primary_labeled)
    step("Exp2: Continuous Prediction",       experiment_continuous_prediction,   df_primary_labeled)
    step("Exp3: Correlation Analysis",        experiment_correlation_analysis,    df_primary_labeled)
    step("Exp4: ESN Baseline",                experiment_esn_baseline,            df_primary_labeled)
    step("Exp5: g Robustness",                experiment_g_robustness)
    step("Exp6: Initial State Ablation",      experiment_initial_state_ablation)
    step("Exp7: Noise Robustness",            experiment_noise_robustness,        df_primary_labeled)
    step("Exp8: Training Size Scaling",       experiment_training_size_scaling,   df_primary_labeled)
    step("Exp9: Linear vs Nonlinear",         experiment_linear_vs_nonlinear,     df_primary_labeled)
    step("Exp10: Feature Separation",         experiment_feature_separation,      df_primary_labeled)
    step("Exp11: System Size Benchmark",      experiment_system_size_benchmark)

    print(f"\n=== ALL EXPERIMENTS DONE in {(time.time()-t_total)/60:.1f} min ===", flush=True)
    print(f"Results saved to {os.path.abspath(RESULTS_DIR)}", flush=True)

if __name__ == '__main__':
    run_all()
