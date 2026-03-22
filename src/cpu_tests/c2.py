import numpy as np
import os
import time
import argparse
import scipy.linalg
import threadpoolctl
import pytaco as pt
from dsl.runner import run
from dsl.var import Var
from dsl.matrix import GeneralMatrix, SymmetricMatrix, DiagonalMatrix
from dsl.operations import opcounters

def verify(name, dsl_val, ref_val):
    error = np.linalg.norm(dsl_val - ref_val) / np.linalg.norm(ref_val)
    print(f"   Verify {name} Error: {error:.2e}")
    return error

def run_taco_step(A_np, B_np, is_diag_A=False, is_diag_B=False):
    try:
        if is_diag_A:
            A_t = pt.from_array(np.diag(A_np).astype(np.float32))
        else:
            A_t = pt.from_array(A_np.astype(np.float32))
        if is_diag_B:
            B_t = pt.from_array(np.diag(B_np).astype(np.float32))
        else:
            B_t = pt.from_array(B_np.astype(np.float32))
        M, K = A_t.shape
        N = B_t.shape[1] if len(B_t.shape) > 1 else 1
        res_t = pt.tensor([M, N], pt.format([pt.dense, pt.dense]))
        i, j, k = pt.get_index_vars(3)
        if len(B_t.shape) > 1:
            res_t[i, j] = A_t[i, k] * B_t[k, j]
        else:
            res_t[i, 0] = A_t[i, k] * B_t[k]
        start = time.time()
        res_t.evaluate()
        return (time.time() - start) * 1000.0
    except:
        return 0.0

def run_straszak_vishnoi_bench(n=4096, m=2048, max_threads=8):
    opcounters.clear()
    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    np.random.seed(42)
    
    tmp = np.random.uniform(-10, 10, (n, m)).astype(np.float32)
    Q, _ = np.linalg.qr(tmp)
    A_full = Q.T.copy() 
    A_T_full = A_full.T.copy() 
    w_diag = np.random.uniform(1, 10, size=n).astype(np.float32) 
    b_vec = np.random.uniform(-10, 10, (m, 1)).astype(np.float32)
    c_vec = np.random.uniform(-10, 10, (n, 1)).astype(np.float32)
    x_curr = np.random.uniform(-10, 10, (n, 1)).astype(np.float32)
    Ax = (A_full @ x_curr).reshape(m, 1)
    rhs_f = b_vec - Ax
    rhs_o = Ax
    h_times, n_times, s_times, t_times, layouts = {}, {}, {}, {}, {}
    n_v, m_v, o_v = Var('n'), Var('m'), Var('one')

    print("\nStep 1: AW = A @ W")
    inputs1 = {'A': A_full, 'W': w_diag, 'M': m, 'N': n, 'K': n, 'n': n, 'm': m, 
               'BM': 128, 'BN': 128, 'BK': 128, 'IT_M': 8, 'IT_N': 8, 'IT_K': 1, 'max_threads': max_threads}
    res, t_ms, _ = run(outs=[GeneralMatrix(name='A', shape=(m_v, n_v)) @ DiagonalMatrix(name='W', shape=(n_v, n_v))],
                        inputs=inputs1, backend="cpp_avx", out_matrix_obj=GeneralMatrix(name='AW', shape=(m_v, n_v)))
    AW_full = res['AW']
    h_times[1] = t_ms
    s=time.time(); AW_ref = A_full @ np.diag(w_diag); n_times[1]=(time.time()-s)*1000.0; s_times[1]=n_times[1]
    t_times[1] = run_taco_step(A_full, w_diag, is_diag_B=True)
    layouts[1] = "Gen @ Diag"
    verify("AW", AW_full, AW_ref)

    print("Step 2: M = AW @ A.T")
    inputs2 = {'AW': AW_full, 'A_T': A_T_full, 'M': m, 'N': m, 'K': n, 'n': n, 'm': m,
               'BM': 128, 'BN': 128, 'BK': 128, 'IT_M': 8, 'IT_N': 8, 'IT_K': 1, 'max_threads': max_threads}
    res, t_ms, _ = run(outs=[GeneralMatrix(name='AW', shape=(m_v, n_v)) @ GeneralMatrix(name='A_T', shape=(n_v, m_v))], 
                        inputs=inputs2, backend="cpp_avx", out_matrix_obj=GeneralMatrix(name='M_raw', shape=(m_v, m_v)))
    M_full = (res['M_raw'] + res['M_raw'].T) / 2 + np.eye(m, dtype=np.float32) * 1e-6 
    h_times[2] = t_ms
    s=time.time(); M_ref = (AW_ref @ A_T_full) + np.eye(m, dtype=np.float32) * 1e-6; n_times[2]=(time.time()-s)*1000.0; s_times[2]=n_times[2]
    t_times[2] = run_taco_step(AW_full, A_T_full)
    layouts[2] = "Gen @ Gen"
    verify("M", M_full, M_ref)

    print("Step 3: Minv = inv(M)")
    s=time.time(); Minv_full = scipy.linalg.inv(M_full.astype(np.float64)).astype(np.float32); h_times[3]=(time.time()-s)*1000.0; n_times[3]=h_times[3]
    s=time.time(); L_c, low = scipy.linalg.cho_factor(M_ref.astype(np.float64)); Minv_cho = scipy.linalg.cho_solve((L_c, low), np.eye(m)); s_times[3]=(time.time()-s)*1000.0
    t_times[3] = 0.0 
    layouts[3] = "SPD Matrix"
    verify("Minv", Minv_full, Minv_cho.astype(np.float32))

    print("Step 4: Q = A.T @ Minv")
    Minv_p = Minv_full[np.triu_indices(m)]
    inputs4 = {'A_T': A_T_full, 'Minv': Minv_p, 'M': n, 'N': m, 'K': m, 'n': n, 'm': m,
               'BM': 128, 'BN': 128, 'BK': 128, 'IT_M': 8, 'IT_N': 8, 'IT_K': 1, 'max_threads': max_threads}
    res, t_ms, _ = run(outs=[GeneralMatrix(name='A_T', shape=(n_v, m_v)) @ SymmetricMatrix(name='Minv', shape=(m_v, m_v))],
                        inputs=inputs4, backend="cpp_avx", out_matrix_obj=GeneralMatrix(name='Q', shape=(n_v, m_v)))
    Q_full = res['Q']
    h_times[4] = t_ms
    s=time.time(); Q_ref = A_T_full @ Minv_full; n_times[4]=(time.time()-s)*1000.0; s_times[4]=n_times[4]
    t_times[4] = run_taco_step(A_T_full, Minv_full)
    layouts[4] = "Gen @ Symm-P"
    verify("Q", Q_full, Q_ref)

    print("Step 5: WQ = W @ Q")
    inputs5 = {'W': w_diag, 'Q': Q_full, 'M': n, 'N': m, 'K': n, 'n': n, 'm': m,
               'BM': 128, 'BN': 128, 'BK': 128, 'IT_M': 8, 'IT_N': 8, 'IT_K': 1, 'max_threads': max_threads}
    res, t_ms, _ = run(outs=[DiagonalMatrix(name='W', shape=(n_v, n_v)) @ GeneralMatrix(name='Q', shape=(n_v, m_v))],
                        inputs=inputs5, backend="cpp_avx", out_matrix_obj=GeneralMatrix(name='WQ', shape=(n_v, m_v)))
    WQ_full = res['WQ']
    h_times[5] = t_ms
    s=time.time(); WQ_ref = np.diag(w_diag) @ Q_ref; n_times[5]=(time.time()-s)*1000.0; s_times[5]=n_times[5]
    t_times[5] = run_taco_step(w_diag, Q_full, is_diag_A=True)
    layouts[5] = "Diag @ Gen"
    verify("WQ", WQ_full, WQ_ref)

    print("Step 6: xf, xo")
    s_f = time.time(); xf = scipy.linalg.blas.sgemv(1.0, WQ_full, rhs_f).reshape(n, 1); t_f = (time.time()-s_f)*1000.0
    s_o = time.time(); xo_p = scipy.linalg.blas.sgemv(1.0, WQ_full, rhs_o).reshape(n, 1); xo = xo_p - (w_diag.reshape(n,1)*c_vec); t_o = (time.time()-s_o)*1000.0
    h_times[6] = t_f + t_o
    s=time.time(); xf_ref = WQ_ref @ rhs_f; xo_ref = WQ_ref @ rhs_o - (w_diag.reshape(n,1)*c_vec); n_times[6]=(time.time()-s)*1000.0; s_times[6]=n_times[6]
    t_times[6] = run_taco_step(WQ_ref, rhs_f) + run_taco_step(WQ_ref, rhs_o)
    layouts[6] = "Gen @ Vector"
    verify("xf", xf, xf_ref); verify("xo", xo, xo_ref)

    lbls = ["AW = A @ W", "M = AW @ A.T", "Minv = inv(M)", "Q = A.T @ Minv", "WQ = W @ Q", "Final xf, xo"]
    print("\n" + "="*145)
    print(f"| {'Step':<20} | {'Layout':<15} | {'DSL (AVX) ms':<12} | {'NumPy ms':<12} | {'TACO ms':<12} | {'SciPy ms':<12} | {'Speedup':<8} |")
    print("-" * 145)
    for i in range(1, 7):
        h, nt, tc, sp, lay = h_times[i], n_times[i], t_times[i], s_times[i], layouts[i]
        tc_s = f"{tc:.4f}" if tc > 0 else "N/A"
        print(f"| {lbls[i-1]:<20} | {lay:<15} | {h:<12.4f} | {nt:<12.4f} | {tc_s:<12} | {sp:<12.4f} | {nt/h:<8.2f}x |")
    print("-" * 145)
    t_h, t_n, t_s, t_t = sum(h_times.values()), sum(n_times.values()), sum(s_times.values()), sum(t_times.values())
    print(f"| {'TOTAL ':<20} | {'-':<15} | {t_h:<12.4f} | {t_n:<12.4f} | {t_t:<12.4f} | {t_s:<12.4f} | {t_n/t_h:<8.2f}x |")
    print("="*145)

if __name__ == '__main__':
    run_straszak_vishnoi_bench()