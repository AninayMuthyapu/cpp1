import torch
import numpy as np
from dsl.var import Var
from dsl.matrix import GeneralMatrix, SymmetricMatrix, DiagonalMatrix
from dsl.runner_cuda import run


def run_pytorch_benchmark(A, B, flops, iters=20):
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)

    start.record()
    for _ in range(iters):
        C = torch.matmul(A, B)
    end.record()

    torch.cuda.synchronize()
    t = start.elapsed_time(end) / iters
    gflops = (flops * 1e-9) / (t / 1000.0)
    return C, t, gflops


def verify(name, custom, torch_res, expected, ct, cg, tt, tg):
    c = custom.cpu().numpy()
    t = torch_res.cpu().numpy()
    e = expected.cpu().numpy()

    print(f"\n{'='*60}")
    print(name)
    print(f"CUDA    : {ct:8.4f} ms | {cg:8.2f} GFLOPS")
    print(f"PyTorch : {tt:8.4f} ms | {tg:8.2f} GFLOPS")

    print("Custom :", "PASS" if np.allclose(c, e, 1e-1, 1e-2) else "FAIL")
    print("Torch  :", "PASS" if np.allclose(t, e, 1e-1, 1e-2) else "FAIL")
    print(f"{'='*60}")


def test_symm_x_gen():
    n_val = 4096
    n = Var("n")

    A_tmp = torch.rand(n_val, n_val, device="cuda")
    A_full = (A_tmp + A_tmp.t()) * 0.5
    idx = torch.triu_indices(n_val, n_val, device="cuda")
    A_pack = A_full[idx[0], idx[1]].contiguous()

    B = torch.rand(n_val, n_val, device="cuda")

    A = SymmetricMatrix((n, n), "A")
    Bm = GeneralMatrix((n, n), "B")
    out = GeneralMatrix((n, n), "C")

    res, ct, cg = run([A @ Bm], {"n": n_val, "A": A_pack, "B": B}, "cuda", out)

    exp = torch.matmul(A_full, B)
    flops = 2.0 * n_val**3
    tr, tt, tg = run_pytorch_benchmark(A_full, B, flops)

    verify("Symm x Gen", res["C"].view(n_val, n_val), tr, exp, ct, cg, tt, tg)


def test_diag_x_gen():
    n_val = 4096
    n = Var("n")

    A_diag = torch.rand(n_val, device="cuda")
    A_full = torch.diag(A_diag)
    B = torch.rand(n_val, n_val, device="cuda")

    A = DiagonalMatrix((n, n), "A")
    Bm = GeneralMatrix((n, n), "B")
    out = GeneralMatrix((n, n), "C")

    res, ct, cg = run([A @ Bm], {"n": n_val, "A": A_diag, "B": B}, "cuda", out)

    exp = torch.matmul(A_full, B)
    flops = 1.0 * n_val**2
    tr, tt, tg = run_pytorch_benchmark(A_full, B, flops)

    verify("Diag x Gen", res["C"].view(n_val, n_val), tr, exp, ct, cg, tt, tg)


def test_symm_x_symm():
    n_val = 4096
    n = Var("n")

    A_tmp = torch.rand(n_val, n_val, device="cuda")
    B_tmp = torch.rand(n_val, n_val, device="cuda")

    A_full = (A_tmp + A_tmp.t()) * 0.5
    B_full = (B_tmp + B_tmp.t()) * 0.5

    idx = torch.triu_indices(n_val, n_val, device="cuda")
    A_pack = A_full[idx[0], idx[1]].contiguous()
    B_pack = B_full[idx[0], idx[1]].contiguous()

    A = SymmetricMatrix((n, n), "A")
    B = SymmetricMatrix((n, n), "B")
    out = GeneralMatrix((n, n), "C")

    res, ct, cg = run([A @ B], {"n": n_val, "A": A_pack, "B": B_pack}, "cuda", out)

    exp = torch.matmul(A_full, B_full)
    flops = 2.0 * n_val**3
    tr, tt, tg = run_pytorch_benchmark(A_full, B_full, flops)

    verify("Symm x Symm", res["C"].view(n_val, n_val), tr, exp, ct, cg, tt, tg)


def test_gen_x_symm():
    n_val = 4096
    n = Var("n")

    A = torch.rand(n_val, n_val, device="cuda")

    B_tmp = torch.rand(n_val, n_val, device="cuda")
    B_full = (B_tmp + B_tmp.t()) * 0.5

    idx = torch.triu_indices(n_val, n_val, device="cuda")
    B_pack = B_full[idx[0], idx[1]].contiguous()

    Am = GeneralMatrix((n, n), "A")
    Bm = SymmetricMatrix((n, n), "B")
    out = GeneralMatrix((n, n), "C")

    res, ct, cg = run([Am @ Bm], {"n": n_val, "A": A, "B": B_pack}, "cuda", out)

    exp = torch.matmul(A, B_full)
    flops = 2.0 * n_val**3
    tr, tt, tg = run_pytorch_benchmark(A, B_full, flops)

    verify("Gen x Symm", res["C"].view(n_val, n_val), tr, exp, ct, cg, tt, tg)


if __name__ == "__main__":
    test_gen_x_symm()
    test_symm_x_gen()
    test_diag_x_gen()
    test_symm_x_symm()
