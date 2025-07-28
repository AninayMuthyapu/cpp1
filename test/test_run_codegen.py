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


def test_13():
    A_t13 = GeneralMatrix((N, N), name="A_t13")
    B_t13 = GeneralMatrix((N, N), name="B_t13") 
    
    C_dsl = A_t13 + B_t13
    D_dsl = C_dsl @ B_t13
    F_dsl = D_dsl - A_t13
    G_dsl = F_dsl + C_dsl
    E_out = G_dsl @ B_t13
    E_out.name = "E_result"
    var_names(None, locals())

    inputs_t13 = {
        "A_t13": np.random.rand(N_val, N_val).astype(np.float32),
        "B_t13": np.random.rand(N_val, N_val).astype(np.float32), 
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([E_out], inputs_t13, backend="cpp")
    E_cpp = res[E_out.name]

    A_np = inputs_t13["A_t13"]
    B_np = inputs_t13["B_t13"]
    
    C_np = A_np + B_np
    D_np = np.matmul(C_np, B_np)
    F_np = D_np - A_np
    G_np = F_np + C_np
    expected_E = np.matmul(G_np, B_np)

    assert np.allclose(E_cpp, expected_E, rtol=rtol, atol=atol)

def test_14():
    M_val_14 = M_val
    N_val_14 = N_val
    P_val_14 = N_val
    K_val_14 = K_val

    A_t14 = GeneralMatrix((M, N), name="A_t14")
    B_t14 = GeneralMatrix((M, N), name="B_t14")
    C_t14 = GeneralMatrix((N, P), name="C_t14")
    D_t14 = GeneralMatrix((M, P), name="D_t14")
    

    Op1 = A_t14 + B_t14
    Op2 = Op1 @ C_t14
    Op3 = Op2 - D_t14
    Out = Op3 + Op2
    Out.name = "Out_result_14"
    var_names(None, locals())

    inputs_t14 = {
        "A_t14": np.random.rand(M_val_14, N_val_14).astype(np.float32),
        "B_t14": np.random.rand(M_val_14, N_val_14).astype(np.float32),
        "C_t14": np.random.rand(N_val_14, P_val_14).astype(np.float32),
        "D_t14": np.random.rand(M_val_14, P_val_14).astype(np.float32),
        "M": M_val_14,
        "N": N_val_14,
        "P": P_val_14,
        "K": K_val_14,
    }

    res = run([Out], inputs_t14, backend="cpp")
    out_cpp = res[Out.name]

    A_np = inputs_t14["A_t14"]
    B_np = inputs_t14["B_t14"]
    C_np = inputs_t14["C_t14"]
    D_np = inputs_t14["D_t14"]

    Op1_np = A_np + B_np
    Op2_np = np.matmul(Op1_np, C_np)
    Op3_np = Op2_np - D_np
    expected_out = Op3_np + Op2_np

    assert np.allclose(out_cpp, expected_out, rtol=rtol, atol=atol)

def test_15():
    A_t15 = GeneralMatrix((M, K), name="A_t15")
    B_t15 = GeneralMatrix((K, N), name="B_t15")
    C_t15 = GeneralMatrix((M, P), name="C_t15")
    D_t15 = GeneralMatrix((P, N), name="D_t15")
    E_t15 = GeneralMatrix((M, N), name="E_t15")
    F_t15 = GeneralMatrix((M, N), name="F_t15")
    G_t15 = GeneralMatrix((M, N), name="G_t15")
    

    Op1 = A_t15 @ B_t15
    Op2 = C_t15 @ D_t15
    Op3 = E_t15 + F_t15
    
    Out = Op1 + Op2 - Op3 + G_t15
    Out.name = "Out_result_15"
    var_names(None, locals())

    inputs_t15 = {
        "A_t15": np.random.rand(M_val, K_val).astype(np.float32),
        "B_t15": np.random.rand(K_val, N_val).astype(np.float32),
        "C_t15": np.random.rand(M_val, P_val).astype(np.float32),
        "D_t15": np.random.rand(P_val, N_val).astype(np.float32),
        "E_t15": np.random.rand(M_val, N_val).astype(np.float32),
        "F_t15": np.random.rand(M_val, N_val).astype(np.float32),
        "G_t15": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_t15, backend="cpp")
    out_cpp = res[Out.name]

    A_np = inputs_t15["A_t15"]
    B_np = inputs_t15["B_t15"]
    C_np = inputs_t15["C_t15"]
    D_np = inputs_t15["D_t15"]
    E_np = inputs_t15["E_t15"]
    F_np = inputs_t15["F_t15"]
    G_np = inputs_t15["G_t15"]

    Op1_np = np.matmul(A_np, B_np)
    Op2_np = np.matmul(C_np, D_np)
    Op3_np = E_np + F_np
    
    expected_out = Op1_np + Op2_np - Op3_np + G_np

    assert np.allclose(out_cpp, expected_out, rtol=rtol, atol=atol)

def test_16():
    A_t16 = GeneralMatrix((N, N), name="A_t16")
    B_t16 = GeneralMatrix((N, N), name="B_t16")
    C_t16 = GeneralMatrix((N, N), name="C_t16")
    D_t16 = GeneralMatrix((N, N), name="D_t16")
    E_t16 = GeneralMatrix((N, N), name="E_t16")
    

    Op1 = A_t16 + (B_t16 @ C_t16)
    Op2 = (Op1 @ Op1) + D_t16
    Out = (Op2 @ Op1) - E_t16
    Out.name = "Out_result_16"
    var_names(None, locals())

    inputs_t16 = {
        "A_t16": np.random.rand(N_val, N_val).astype(np.float32),
        "B_t16": np.random.rand(N_val, N_val).astype(np.float32),
        "C_t16": np.random.rand(N_val, N_val).astype(np.float32),
        "D_t16": np.random.rand(N_val, N_val).astype(np.float32),
        "E_t16": np.random.rand(N_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_t16, backend="cpp")
    out_cpp = res[Out.name]

    A_np = inputs_t16["A_t16"]
    B_np = inputs_t16["B_t16"]
    C_np = inputs_t16["C_t16"]
    D_np = inputs_t16["D_t16"]
    E_np = inputs_t16["E_t16"]

    Op1_np = A_np + np.matmul(B_np, C_np)
    Op2_np = np.matmul(Op1_np, Op1_np) + D_np
    expected_out = np.matmul(Op2_np, Op1_np) - E_np

    assert np.allclose(out_cpp, expected_out, rtol=rtol, atol=atol)

def test_17():
    A_t17 = GeneralMatrix((N, N), name="A_t17")
    B_t17 = GeneralMatrix((N, N), name="B_t17")
    C_t17 = GeneralMatrix((N, N), name="C_t17")
    D_t17 = GeneralMatrix((N, N), name="D_t17")
    E_t17 = GeneralMatrix((N, N), name="E_t17")
    F_t17 = GeneralMatrix((N, N), name="F_t17")
    G_t17 = GeneralMatrix((N, N), name="G_t17")
    H_t17 = GeneralMatrix((N, N), name="H_t17")
    

    Op1_intermediate = (A_t17 @ B_t17) + (C_t17 - D_t17)
    Op2_intermediate = (Op1_intermediate @ E_t17) - (F_t17 + G_t17)
    Out = Op2_intermediate + (Op1_intermediate @ H_t17)
    Out.name = "Out_result_17"
    var_names(None, locals())

    inputs_t17 = {
        "A_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "B_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "C_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "D_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "E_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "F_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "G_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "H_t17": np.random.rand(N_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_t17, backend="cpp")
    out_cpp = res[Out.name]

    A_np = inputs_t17["A_t17"]
    B_np = inputs_t17["B_t17"]
    C_np = inputs_t17["C_t17"]
    D_np = inputs_t17["D_t17"]
    E_np = inputs_t17["E_t17"]
    F_np = inputs_t17["F_t17"]
    G_np = inputs_t17["G_t17"]
    H_np = inputs_t17["H_t17"]

    Op1_intermediate_np = np.matmul(A_np, B_np) + (C_np - D_np)
    Op2_intermediate_np = np.matmul(Op1_intermediate_np, E_np) - (F_np + G_np)
    expected_out = Op2_intermediate_np + np.matmul(Op1_intermediate_np, H_np)

    assert np.allclose(out_cpp, expected_out, rtol=rtol, atol=atol)

def test_18():
    A_t18 = GeneralMatrix((M, K), name="A_t18")
    B_t18 = GeneralMatrix((K, N), name="B_t18")
    C_t18 = GeneralMatrix((M, N), name="C_t18")
    D_t18 = GeneralMatrix((M, N), name="D_t18")
    E_t18 = GeneralMatrix((N, P), name="E_t18")
    F_t18 = GeneralMatrix((M, P), name="F_t18")
    G_t18 = GeneralMatrix((M, P), name="G_t18")
    

    Op1 = A_t18 @ B_t18
    Op2 = C_t18 + D_t18
    Op3 = Op1 - Op2
    Op4 = Op3 @ E_t18
    Out = Op4 + F_t18 - G_t18
    Out.name = "Out_result_18"
    var_names(None, locals())

    inputs_t18 = {
        "A_t18": np.random.rand(M_val, K_val).astype(np.float32),
        "B_t18": np.random.rand(K_val, N_val).astype(np.float32),
        "C_t18": np.random.rand(M_val, N_val).astype(np.float32),
        "D_t18": np.random.rand(M_val, N_val).astype(np.float32),
        "E_t18": np.random.rand(N_val, P_val).astype(np.float32),
        "F_t18": np.random.rand(M_val, P_val).astype(np.float32),
        "G_t18": np.random.rand(M_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs_t18, backend="cpp")
    out_cpp = res[Out.name]

    A_np = inputs_t18["A_t18"]
    B_np = inputs_t18["B_t18"]
    C_np = inputs_t18["C_t18"]
    D_np = inputs_t18["D_t18"]
    E_np = inputs_t18["E_t18"]
    F_np = inputs_t18["F_t18"]
    G_np = inputs_t18["G_t18"]

    Op1_np = np.matmul(A_np, B_np)
    Op2_np = C_np + D_np
    Op3_np = Op1_np - Op2_np
    Op4_np = np.matmul(Op3_np, E_np)
    expected_out = Op4_np + F_np - G_np

    assert np.allclose(out_cpp, expected_out, rtol=rtol, atol=atol)