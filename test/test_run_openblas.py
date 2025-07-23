import pytest
import numpy as np
from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.runner import run
from dsl.utils import var_names

M = Var("M")
N = Var("N")
P = Var("P") 

M_val = 128
N_val = 64
P_val = 32

rtol = 1e-5
atol = 1e-5

def test_add_op_openblas():
    

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
    }
    
    res = run([C_out], temp_inputs, backend="openblas")
    C_openblas = res[C_out.name]

    expected_C = temp_inputs["TempA"] + temp_inputs["TempB"]
    
   

    assert np.allclose(C_openblas, expected_C, rtol=rtol, atol=atol)
   
def test_matmul_openblas():
   

    B_global = GeneralMatrix((M, N), name="B_global")
    C_global = GeneralMatrix((N, P), name="C_global")
    var_names(None, locals())

    Out = B_global @ C_global
    Out.name = "Out_result"

    inputs_matmul = {
        "B_global": np.random.rand(M_val, N_val).astype(np.float32),
        "C_global": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
    }

    res = run([Out], inputs_matmul, backend="openblas")
    Out_openblas = res[Out.name]

    expected_Out = np.matmul(inputs_matmul["B_global"], inputs_matmul["C_global"])


    assert np.allclose(Out_openblas, expected_Out, rtol=1e-05, atol=1e-05)


def test_add_matmul_openblas():
    
    A_global = GeneralMatrix((M, P), name="A_global")
    B_global = GeneralMatrix((M, N), name="B_global")
    C_global = GeneralMatrix((N, P), name="C_global")
    var_names(None, locals())

    Out = A_global + (B_global @ C_global)
    Out.name = "Out_result"

    inputs_add_matmul = {
        "A_global": np.random.rand(M_val, P_val).astype(np.float32),
        "B_global": np.random.rand(M_val, N_val).astype(np.float32),
        "C_global": np.random.rand(N_val, P_val).astype(np.float32),
        "M": M_val,
        "N": N_val,
        "P": P_val,
    }

    res = run([Out], inputs_add_matmul, backend="openblas")
    Out_openblas = res[Out.name]
    
    expected_B_C = np.matmul(inputs_add_matmul["B_global"], inputs_add_matmul["C_global"])
    expected_Out = inputs_add_matmul["A_global"] + expected_B_C


    assert np.allclose(Out_openblas, expected_Out, rtol=rtol, atol=atol)
    

