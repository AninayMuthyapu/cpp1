import pytest
import numpy as np
from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.runner import run
from dsl.utils import var_names

M = Var("M")
N = Var("N")

A = GeneralMatrix((M, N))
B = GeneralMatrix((M, N))
E = GeneralMatrix((M, N))

var_names(None, locals())

M_val = 32
N_val = M_val

inputs = {
    "A": np.random.rand(M_val, N_val).astype(np.float32),
    "B": np.random.rand(M_val, N_val).astype(np.float32),
    "E": np.random.rand(M_val, N_val).astype(np.float32),
    "M": M_val,
    "N": N_val,
}

rtol = 1e-5
atol = 1e-5

def test_add_op():
    C_out = A + B
    C_out.name = "C_result"
    res = run([C_out], inputs, backend="numpy")
    C_np = res[C_out.name]
    expected_C = inputs["A"] + inputs["B"]
    assert np.allclose(C_np, expected_C, rtol=rtol, atol=atol)

def test_mul_add_op():
    C_intermediate = A + B
    D_out = (C_intermediate @ B) + E
    D_out.name = "D_result"
    res = run([D_out], inputs, backend="numpy")
    D_np = res[D_out.name]
    expected_C_intermediate = inputs["A"] + inputs["B"]
    expected_D = np.matmul(expected_C_intermediate, inputs["B"]) + inputs["E"]
    assert np.allclose(D_np, expected_D, rtol=rtol, atol=atol)

def test_final_sub_op():
    C_intermediate = A + B
    D_intermediate = (C_intermediate @ B) + E
    F_out = D_intermediate - C_intermediate
    F_out.name = "F_result"
    res = run([F_out], inputs, backend="numpy")
    F_np = res[F_out.name]
    expected_C_intermediate = inputs["A"] + inputs["B"]
    expected_D_intermediate = np.matmul(expected_C_intermediate, inputs["B"]) + inputs["E"]
    expected_F = expected_D_intermediate - expected_C_intermediate
    assert np.allclose(F_np, expected_F, rtol=rtol, atol=atol)

def test_final_sub_op2():
    C_intermediate = A + B
    D_intermediate = (C_intermediate @ B) + E
    F_out = D_intermediate - C_intermediate
    F_out.name = "F_result"
    res = run([F_out], inputs, backend="numpy")
    F_np = res[F_out.name]
    expected_C_intermediate = inputs["A"] + inputs["B"]
    expected_D_intermediate = np.matmul(expected_C_intermediate, inputs["B"]) + inputs["E"]
    expected_F = expected_D_intermediate - expected_C_intermediate
    assert np.allclose(F_np, expected_F, rtol=rtol, atol=atol)
