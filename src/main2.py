import numpy as np
import torch
import time

from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.runner_cuda import run

def run_pytorch_benchmark(n, A_np, B_np):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A_torch = torch.from_numpy(A_np).to(device)
    B_torch = torch.from_numpy(B_np).to(device)

    iters = 20
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        C_torch = torch.matmul(A_torch, B_torch)
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) / iters

    flops = 2.0 * (n ** 3)
    gflops = (flops * 1e-9) / (elapsed_time_ms / 1000.0)

    return C_torch.cpu().numpy(), elapsed_time_ms, gflops

def test_gen_x_gen_comparison():
    n_val = 4096 
    
    A_full = np.random.rand(n_val, n_val).astype(np.float32)
    B_full = np.random.rand(n_val, n_val).astype(np.float32)

    n = Var("n")
    A = GeneralMatrix((n, n), "A")
    B = GeneralMatrix((n, n), "B")
    op_node = A @ B 
    out_matrix = GeneralMatrix((n, n), "C")

    input_data = {
        "n": n_val,
        "A": A_full,
        "B": B_full 
    }

    results, custom_time_ms, custom_gflops = run(
        outs=[op_node], 
        inputs=input_data, 
        backend="cuda", 
        out_matrix_obj=out_matrix
    )
    custom_result = results["C"]

    torch_result, torch_time_ms, torch_gflops = run_pytorch_benchmark(n_val, A_full, B_full)

    print(f" CUDA - Time: {custom_time_ms:.4f} ms | GFLOPS: {custom_gflops:.4f}")
    print(f"PyTorch     - Time: {torch_time_ms:.4f} ms | GFLOPS: {torch_gflops:.4f}")
    
    expected = A_full @ B_full
    
    if np.allclose(custom_result, expected, atol=1e-2, rtol=1e-3):
        print(" Output matches.")
    else:
        diff = np.max(np.abs(custom_result - expected))
        print(f" Output mismatch (Max Diff: {diff:.6f})")

    if np.allclose(torch_result, expected, atol=1e-2, rtol=1e-3):
        print("PyTorch Output matches .")
    else:
        diff = np.max(np.abs(torch_result - expected))
        print(f"PyTorch Output mismatch (Max Diff: {diff:.6f})")

if __name__ == "__main__":
    test_gen_x_gen_comparison()