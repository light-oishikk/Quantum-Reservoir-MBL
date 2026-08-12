import numpy as np
from scipy.linalg import eig
import joblib
from joblib import Parallel, delayed
import time
import warnings

# =============================================================================
# 1. Spin operator helpers
# =============================================================================

I2 = np.eye(2, dtype=np.complex128)
sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
sp = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
sm = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)

def op_on_site(op, site, N):
    """
    Place a 2x2 operator on site `site` in an N-site chain via tensor products.
    """
    if site < 0 or site >= N:
        raise ValueError("Site index out of bounds.")
    
    ops = [I2] * N
    ops[site] = op
    
    res = ops[0]
    for i in range(1, N):
        res = np.kron(res, ops[i])
        
    return res

# =============================================================================
# 2. Hamiltonian construction
# =============================================================================

def build_H_NH(h, N, g, J=1.0, Delta=1.0, pbc=True):
    """
    Builds the non-Hermitian XXZ Hamiltonian.
    H = sum_<i,j> [J*exp(g) S+_i S-_j + J*exp(-g) S-_i S+_j + J*Delta Sz_i Sz_j] 
      + sum_i h_i Sz_i
    
    Returns:
        H: The Hamiltonian matrix (2^N x 2^N).
        Sz_ops: List of Sz operators for each site.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=np.complex128)
    
    # Create Sz operators
    Sz_ops = [op_on_site(sz, i, N) for i in range(N)]
    
    # Disorder terms
    for i in range(N):
        H += h[i] * Sz_ops[i]
        
    # Interaction terms
    for i in range(N):
        if not pbc and i == N - 1:
            continue
        j = (i + 1) % N
        
        Sp_i = op_on_site(sp, i, N)
        Sm_j = op_on_site(sm, j, N)
        
        Sm_i = op_on_site(sm, i, N)
        Sp_j = op_on_site(sp, j, N)
        
        Sz_i = Sz_ops[i]
        Sz_j = Sz_ops[j]
        
        H += J * np.exp(g) * (Sp_i @ Sm_j)
        H += J * np.exp(-g) * (Sm_i @ Sp_j)
        H += J * Delta * (Sz_i @ Sz_j)
        
    return H, Sz_ops

# =============================================================================
# 3. Symmetry sector
# =============================================================================

def sector_indices(N):
    """
    Returns indices of basis states in the Sz=0 sector.
    """
    indices = []
    for i in range(2**N):
        # Count the number of set bits (spins up)
        n_up = bin(i).count("1")
        sz_val = 2 * n_up - N
        if sz_val == 0:
            indices.append(i)
    return np.array(indices)

def make_product_state_index(bits):
    """
    General product state index from bit pattern.
    """
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx

def neel_index(N):
    """
    Returns the basis index of the Neel state |uparrow downarrow uparrow downarrow...>.
    """
    bits = [(i + 1) % 2 for i in range(N)]
    return make_product_state_index(bits)

# =============================================================================
# 4. Single realization computation
# =============================================================================

def compute_single_realization(
    W, N, g, rng_seed, 
    meas_sites=None, meas_times=(0,1,2,4,8),
    include_survival=True, cond_threshold=1e10,
    initial_state_idx=None
):
    """
    Compute features and labels for a single disorder realization.
    """
    rng = np.random.default_rng(rng_seed)
    h = rng.uniform(-W, W, size=N)
    
    if meas_sites is None:
        meas_sites = list(range(N))
        
    H, Sz_ops = build_H_NH(h, N, g)
    
    # f_real in Sz=0 sector
    idx_0 = sector_indices(N)
    H_0 = H[np.ix_(idx_0, idx_0)]
    try:
        evals_0 = np.linalg.eigvals(H_0)
    except np.linalg.LinAlgError:
        return None
        
    f_real = np.mean(np.abs(np.imag(evals_0)) < 1e-10)
    
    # Full diagonalization
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            evals, evecs = eig(H)
        except Exception:
            return None
            
    cond = np.linalg.cond(evecs)
    if cond > cond_threshold or np.isinf(cond):
        return None
        
    if initial_state_idx is None:
        initial_state_idx = neel_index(N)
        
    psi0 = np.zeros(2**N, dtype=np.complex128)
    psi0[initial_state_idx] = 1.0
    
    try:
        c = np.linalg.solve(evecs, psi0)
    except np.linalg.LinAlgError:
        return None
        
    features = []
    
    for t in meas_times:
        psi_t = evecs @ (c * np.exp(-1j * evals * t))
        
        norm_sq = np.real(np.vdot(psi_t, psi_t))
        if norm_sq < 1e-30:
            norm_sq = 1e-30
            
        t_feats = []
        if include_survival:
            t_feats.append(np.log(norm_sq))
            
        for s in meas_sites:
            Sz_s = Sz_ops[s]
            val = np.real(np.vdot(psi_t, Sz_s @ psi_t)) / norm_sq
            t_feats.append(val)
            
        features.extend(t_feats)
        
    oracle_features = [float(np.mean(h)), float(np.std(h))]
    
    return {
        'features': features,
        'oracle_features': oracle_features,
        'f_real': f_real,
        'W': W,
        'skipped': False
    }

# =============================================================================
# 5. Parallelized dataset generation
# =============================================================================

def build_dataset_parallel(
    N=10, g=0.3, 
    W_list=(0.2, 0.5, 1, 1.5, 2, 3, 4, 6, 10),
    n_real=500, seed=0,
    meas_sites=None, meas_times=(0,1,2,4,8),
    include_survival=True, cond_threshold=1e10,
    initial_state_idx=None,
    n_jobs=-1, verbose=1
):
    tasks = []
    for w_idx, W in enumerate(W_list):
        for r in range(n_real):
            # Unique deterministic seed for each realization
            rng_seed = abs(hash((seed, w_idx, r))) % (2**32 - 1)
            tasks.append((W, N, g, rng_seed, meas_sites, meas_times, include_survival, cond_threshold, initial_state_idx))
            
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_single_realization)(*task) for task in tasks
    )
    
    X_features = []
    X_oracle = []
    y_freal = []
    y_W = []
    skipped = 0
    
    for res in results:
        if res is None:
            skipped += 1
            continue
        X_features.append(res['features'])
        X_oracle.append(res['oracle_features'])
        y_freal.append(res['f_real'])
        y_W.append(res['W'])
        
    return np.array(X_features), np.array(X_oracle), np.array(y_freal), np.array(y_W), skipped

# =============================================================================
# 6. W* estimation
# =============================================================================

def estimate_W_star(y_freal, y_W):
    """
    Compute the pseudo-critical disorder W* where mean f_real crosses 0.5.
    """
    W_vals = np.sort(np.unique(y_W))
    mean_freal = [np.mean(y_freal[y_W == w]) for w in W_vals]
    
    W_star = None
    for i in range(len(W_vals) - 1):
        f1, f2 = mean_freal[i], mean_freal[i+1]
        w1, w2 = W_vals[i], W_vals[i+1]
        
        if (f1 - 0.5) * (f2 - 0.5) <= 0:
            if f1 != f2:
                W_star = w1 + (0.5 - f1) * (w2 - w1) / (f2 - f1)
            else:
                W_star = (w1 + w2) / 2.0
            break
            
    if W_star is None:
        W_star = np.mean(W_vals)
        
    return W_star

# =============================================================================
# 7. Phase labeling
# =============================================================================

def assign_phase_labels(y_W, W_star):
    """
    Returns binary labels: 0 if W <= W*, 1 if W > W*
    """
    return (y_W > W_star).astype(int)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Testing NH MBL Core Engine...")
    t0 = time.time()
    
    X, X_oracle, y_f, y_w, skipped = build_dataset_parallel(
        N=8, g=0.1, 
        W_list=(0.5, 2.0, 4.0),
        n_real=10, seed=42,
        meas_times=(0,1,2),
        n_jobs=-1, verbose=10
    )
    
    t1 = time.time()
    
    print(f"Elapsed time: {t1 - t0:.2f} s")
    print(f"Features shape: {X.shape}")
    print(f"Oracle features shape: {X_oracle.shape}")
    print(f"Skipped realizations: {skipped}")
    
    W_star = estimate_W_star(y_f, y_w)
    print(f"Estimated W*: {W_star:.2f}")
    
    labels = assign_phase_labels(y_w, W_star)
    print(f"Labels shape: {labels.shape}, phase 1 fraction: {labels.mean():.2f}")
