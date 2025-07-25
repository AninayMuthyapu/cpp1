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
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    var_names(None, locals())

    Out = A + B
    Out.name = "Out_result"

    inputs = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }
    
    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = inputs["A"] + inputs["B"]
    
    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_2():
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    var_names(None, locals())

    Out = B @ C
    Out.name = "Out_result"

    inputs = {
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = np.matmul(inputs["B"], inputs["C"])

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_3():
    A = GeneralMatrix((M, P), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    var_names(None, locals())

    Out = A + (B @ C)
    Out.name = "Out_result"

    inputs = {
        "A": np.random.rand(M_val, P_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]
    
    expected_out = inputs["A"] + (inputs["B"] @ inputs["C"])

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_4():
    A = GeneralMatrix((M, K), name="A")
    B = GeneralMatrix((K, N), name="B")
    C = GeneralMatrix((M, N), name="C")
    var_names(None, locals())

    Out = (A @ B) - C
    Out.name = "Out_result"

    inputs = {
        "A": np.random.rand(M_val, K_val).astype(np.float32),
        "B": np.random.rand(K_val, N_val).astype(np.float32),
        "C": np.random.rand(M_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = (inputs["A"] @ inputs["B"]) - inputs["C"]

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_5():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, K), name="B")
    C = GeneralMatrix((K, N), name="C")
    D = GeneralMatrix((M, N), name="D")
    E = GeneralMatrix((M, N), name="E")
    var_names(None, locals())

    Out = A + (B @ C) - (D + E)
    Out.name = "Out_result"

    inputs = {
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

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = inputs["A"] + (inputs["B"] @ inputs["C"]) - (inputs["D"] + inputs["E"])

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_6():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    D = GeneralMatrix((N, P), name="D")
    var_names(None, locals())

    Out = (A + B) @ (C - D)
    Out.name = "Out_result"

    inputs = {
        "A": np.random.rand(M_val, N_val).astype(np.float32),
        "B": np.random.rand(M_val, N_val).astype(np.float32),
        "C": np.random.rand(N_val, P_val).astype(np.float32),
        "D": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = (inputs["A"] + inputs["B"]) @ (inputs["C"] - inputs["D"])

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_7():
    A = GeneralMatrix((M, K), name="A")
    B = GeneralMatrix((K, N), name="B")
    C = GeneralMatrix((M, P), name="C")
    D = GeneralMatrix((P, N), name="D")
    var_names(None, locals())

    Out = (A @ B) + (C @ D)
    Out.name = "Out_result"

    inputs = {
        "A": np.random.rand(M_val, K_val).astype(np.float32),
        "B": np.random.rand(K_val, N_val).astype(np.float32),
        "C": np.random.rand(M_val, P_val).astype(np.float32),
        "D": np.random.rand(P_val, N_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
        "K": K_val,
    }

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = (inputs["A"] @ inputs["B"]) + (inputs["C"] @ inputs["D"])

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


def test_8():
    A = GeneralMatrix((M, N), name="A")
    B = GeneralMatrix((M, N), name="B")
    C = GeneralMatrix((N, P), name="C")
    D = GeneralMatrix((M, P), name="D")
    E = GeneralMatrix((M, P), name="E")
    var_names(None, locals())

    Out = (A + B) @ C + D - E
    Out.name = "Out_result"

    inputs = {
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

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = ((inputs["A"] + inputs["B"]) @ inputs["C"]) + inputs["D"] - inputs["E"]

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


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

    inputs = {
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

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = (inputs["A"] @ inputs["B"]) + \
                   (inputs["C"] @ inputs["D"]) - \
                   (inputs["E"] + inputs["F"])

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


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

    inputs = {
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

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = ((inputs["A"] + inputs["B"]) @ inputs["C"]) - \
                   (inputs["D"] @ inputs["E"]) + \
                   inputs["F"]

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


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

    inputs = {
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

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = (((inputs["A"] + inputs["B"]) @ inputs["C"]) - \
                   (inputs["D"] + inputs["E"])) @ inputs["F"] + \
                   inputs["G"] - inputs["H"] + inputs["I"]

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)


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

    inputs = {
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

    res = run([Out], inputs, backend="openblas")
    out_blas = res[Out.name]

    expected_out = ((((inputs["A"] @ inputs["B"]) + \
                      (inputs["C"] @ inputs["D"])) - \
                     (inputs["E"] @ inputs["F"])) @ inputs["G"]) + \
                   inputs["H"] - inputs["I"] + inputs["J"]

    assert np.allclose(out_blas, expected_out, rtol=rtol, atol=atol)

# from dsl.var import Var
# from dsl.matrix import GeneralMatrix
# from dsl.utils import topological_sort_operations, var_names

# # Define symbolic variables for matrix shapes
# M = Var("M")
# N = Var("N")
# P = Var("P")

# # Define input matrices
# A_global = GeneralMatrix((M, P), name="A_global")
# B_global = GeneralMatrix((M, N), name="B_global")
# C_global = GeneralMatrix((N, P), name="C_global")

# # Assign names to symbolic matrices from the local scope.
# var_names(None, locals())

# # Define the DSL expression directly in one line
# # The (B_global @ C_global) part will implicitly create a MatMul Operation object.
# Out = A_global + (B_global @ C_global)
# Out.name = "Out_Result" # Name the final output operation

# # Perform topological sort on the final output node
# sorted_ops = topological_sort_operations([Out])

# print("Topologically Sorted Operations for 'Out = A_global + (B_global @ C_global)' (direct expression):")
# for op in sorted_ops:
#     print(f"- {op.name} (Type: {op.operations_type})")