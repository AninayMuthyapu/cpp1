import pytest
import numpy as np
from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.runner import run
from dsl.utils import var_names

M = Var("M")
N = Var("N")
P = Var("P")
K = Var("K")

M_val = 128
N_val = 64
P_val = 32
K_val = 96

rtol = 1e-5
atol = 1e-5

def test_1():
    TempA = GeneralMatrix((M, N), name="TempA")
    TempB = GeneralMatrix((M, N), name="TempB")
    var_names(None, locals())

    C_out = TempA + TempB
    C_out.name = "C_result"

    temp_inputs = {
        "TempA": np.random.rand(M_val, N_val).astype(np.float32),
        "TempB": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }
    
    res = run([C_out], temp_inputs, backend="cpp")
    C_cpp = res[C_out.name]

    expected_C = temp_inputs["TempA"] + temp_inputs["TempB"]
    
    assert np.allclose(C_cpp, expected_C, rtol=rtol, atol=atol)


def test_2():
    B = GeneralMatrix((M, N), name="B_global")
    C = GeneralMatrix((N, P), name="C_global")
    var_names(None, locals())

    Out = B @ C
    Out.name = "Out_result"

    inputs_matmul = {
        "B_global": np.random.rand(M_val, N_val).astype(np.float32),
        "C_global": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_matmul, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = np.matmul(inputs_matmul["B_global"], inputs_matmul["C_global"])

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_3():
    A = GeneralMatrix((M, P), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    var_names(None, locals())

    Out = A + (B @ C)
    Out.name = "Out_result"

    inputs_add_matmul = {
        "A": np.random.rand(M_val, P_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_add_matmul, backend="cpp")
    Out_cpp = res[Out.name]
    
    expected_Out = inputs_add_matmul["A"] + (inputs_add_matmul["B"] @ inputs_add_matmul["C"])

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_4():
    A = GeneralMatrix((M, K), name="A")
    B = GeneralMatrix((K, N), name="B")
    C = GeneralMatrix((M, N), name="C")
    var_names(None, locals())

    Out = (A @ B) - C
    Out.name = "Out_result"

    inputs_matmul_sub = {
        "A": np.random.rand(M_val, K_val).astype(np.float32),
        "B": np.random.rand(K_val, N_val).astype(np.float32),
        "C": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_matmul_sub, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = (inputs_matmul_sub["A"] @ inputs_matmul_sub["B"]) - inputs_matmul_sub["C"]

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_5():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, K), name="B")
    C = GeneralMatrix((K, N), name="C")
    D = GeneralMatrix((M, N), name="D")
    E = GeneralMatrix((M, N), name="E")
    var_names(None, locals())

    Out = A + (B @ C) - (D + E)
    Out.name = "Out_result"

    inputs_complex_expr_1 = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, K_val).astype(np.float32),
        "C": np.random.rand(K_val, N_val).astype(np.float32),
        "D": np.random.rand(M_val, N_val).astype(np.float32),
        "E": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_1, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = inputs_complex_expr_1["A"] + (inputs_complex_expr_1["B"] @ inputs_complex_expr_1["C"]) - (inputs_complex_expr_1["D"] + inputs_complex_expr_1["E"])

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_6():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    D = GeneralMatrix((N, P), name="D")
    var_names(None, locals())

    Out = (A + B) @ (C - D)
    Out.name = "Out_result"

    inputs_complex_expr_2 = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "D": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_2, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = (inputs_complex_expr_2["A"] + inputs_complex_expr_2["B"]) @ (inputs_complex_expr_2["C"] - inputs_complex_expr_2["D"])

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_7():
    A = GeneralMatrix((M, K), name="A")
    B = GeneralMatrix((K, N), name="B")
    C = GeneralMatrix((M, P), name="C")
    D = GeneralMatrix((P, N), name="D")
    var_names(None, locals())

    Out = (A @ B) + (C @ D)
    Out.name = "Out_result"

    inputs_complex_expr_3 = {
        "A": np.random.rand(M_val, K_val).astype(np.float32),
        "B": np.random.rand(K_val, N_val).astype(np.float32),
        "C": np.random.rand(M_val, P_val).astype(np.float32),
        "D": np.random.rand(P_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_3, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = (inputs_complex_expr_3["A"] @ inputs_complex_expr_3["B"]) + (inputs_complex_expr_3["C"] @ inputs_complex_expr_3["D"])

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_8():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    D = GeneralMatrix((M, P), name="D")
    E = GeneralMatrix((M, P), name="E")
    var_names(None, locals())

    Out = (A + B) @ C + D - E
    Out.name = "Out_result"

    inputs_complex_expr_4 = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "D": np.random.rand(M_val, P_val).astype(np.float32),
        "E": np.random.rand(M_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_4, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = ((inputs_complex_expr_4["A"] + inputs_complex_expr_4["B"]) @ inputs_complex_expr_4["C"]) + inputs_complex_expr_4["D"] - inputs_complex_expr_4["E"]

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_9():
    A = GeneralMatrix((M, K), name="A")
    B = GeneralMatrix((K, N), name="B")
    C = GeneralMatrix((M, P), name="C")
    D = GeneralMatrix((P, N), name="D")
    E = GeneralMatrix((M, N), name="E")
    F = GeneralMatrix((M, N), name="F")
    var_names(None, locals())

    Out = (A @ B) + (C @ D) - (E + F)
    Out.name = "Out_result"

    inputs_complex_expr_5 = {
        "A": np.random.rand(M_val, K_val).astype(np.float32),
        "B": np.random.rand(K_val, N_val).astype(np.float32),
        "C": np.random.rand(M_val, P_val).astype(np.float32),
        "D": np.random.rand(P_val, N_val).astype(np.float32),
        "E": np.random.rand(M_val, N_val).astype(np.float32),
        "F": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_5, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = (inputs_complex_expr_5["A"] @ inputs_complex_expr_5["B"]) + \
                   (inputs_complex_expr_5["C"] @ inputs_complex_expr_5["D"]) - \
                   (inputs_complex_expr_5["E"] + inputs_complex_expr_5["F"])

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_10():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    D = GeneralMatrix((M, K), name="D")
    E = GeneralMatrix((K, P), name="E")
    F = GeneralMatrix((M, P), name="F")
    var_names(None, locals())

    Out = ((A + B) @ C) - (D @ E) + F
    Out.name = "Out_result"

    inputs_complex_expr_6 = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "D": np.random.rand(M_val, K_val).astype(np.float32),
        "E": np.random.rand(K_val, P_val).astype(np.float32),
        "F": np.random.rand(M_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_6, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = ((inputs_complex_expr_6["A"] + inputs_complex_expr_6["B"]) @ inputs_complex_expr_6["C"]) - \
                   (inputs_complex_expr_6["D"] @ inputs_complex_expr_6["E"]) + \
                   inputs_complex_expr_6["F"]

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_11():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    D = GeneralMatrix((M, P), name="D")
    E = GeneralMatrix((M, P), name="E")
    F = GeneralMatrix((P, K), name="F")
    G = GeneralMatrix((M, K), name="G")
    H = GeneralMatrix((M, K), name="H")
    I = GeneralMatrix((M, K), name="I")
    var_names(None, locals())

    Out = (((A + B) @ C) - (D + E)) @ F + G - H + I
    Out.name = "Out_result"

    inputs_complex_expr_7 = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "D": np.random.rand(M_val, P_val).astype(np.float32),
        "E": np.random.rand(M_val, P_val).astype(np.float32),
        "F": np.random.rand(P_val, K_val).astype(np.float32),
        "G": np.random.rand(M_val, K_val).astype(np.float32),
        "H": np.random.rand(M_val, K_val).astype(np.float32),
        "I": np.random.rand(M_val, K_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_7, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = (((inputs_complex_expr_7["A"] + inputs_complex_expr_7["B"]) @ inputs_complex_expr_7["C"]) - \
                   (inputs_complex_expr_7["D"] + inputs_complex_expr_7["E"])) @ inputs_complex_expr_7["F"] + \
                   inputs_complex_expr_7["G"] - inputs_complex_expr_7["H"] + inputs_complex_expr_7["I"]

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)


def test_12():
    A = GeneralMatrix((M, K), name="A")
    B = GeneralMatrix((K, N), name="B")
    C = GeneralMatrix((M, K), name="C")
    D = GeneralMatrix((K, N), name="D")
    E = GeneralMatrix((M, K), name="E")
    F = GeneralMatrix((K, N), name="F")
    G = GeneralMatrix((N, P), name="G")
    H = GeneralMatrix((M, P), name="H")
    I = GeneralMatrix((M, P), name="I")
    J = GeneralMatrix((M, P), name="J")
    var_names(None, locals())

    Out = ((((A @ B) + (C @ D)) - (E @ F)) @ G) + H - I + J
    Out.name = "Out_result"

    inputs_complex_expr_8 = {
        "A": np.random.rand(M_val, K_val).astype(np.float32),
        "B": np.random.rand(K_val, N_val).astype(np.float32),
        "C": np.random.rand(M_val, K_val).astype(np.float32),
        "D": np.random.rand(K_val, N_val).astype(np.float32),
        "E": np.random.rand(M_val, K_val).astype(np.float32),
        "F": np.random.rand(K_val, N_val).astype(np.float32),
        "G": np.random.rand(N_val, P_val).astype(np.float32),
        "H": np.random.rand(M_val, P_val).astype(np.float32),
        "I": np.random.rand(M_val, P_val).astype(np.float32),
        "J": np.random.rand(M_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_complex_expr_8, backend="cpp")
    Out_cpp = res[Out.name]

    expected_Out = ((((inputs_complex_expr_8["A"] @ inputs_complex_expr_8["B"]) + \
                      (inputs_complex_expr_8["C"] @ inputs_complex_expr_8["D"])) - \
                     (inputs_complex_expr_8["E"] @ inputs_complex_expr_8["F"])) @ inputs_complex_expr_8["G"]) + \
                   inputs_complex_expr_8["H"] - inputs_complex_expr_8["I"] + inputs_complex_expr_8["J"]

    assert np.allclose(Out_cpp, expected_Out, rtol=rtol, atol=atol)