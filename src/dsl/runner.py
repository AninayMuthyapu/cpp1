import ctypes
import numpy as np
import os
from dsl.var import Var  


def run(outputs, inputs: dict):
    def resolve_dim(dim):
        if isinstance(dim, Var):
            return int(dim.name)  
        return dim

    
    lib_path = os.path.abspath("src/cpp/libdsl.so")
    lib = ctypes.CDLL(lib_path)

    func = lib.compute
    func.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3 + [ctypes.c_int, ctypes.c_int]

    output = outputs[0]
    M = resolve_dim(output.shape[0])
    N = resolve_dim(output.shape[1])

    A_np = list(inputs.values())[0].astype(np.float32)
    B_np = list(inputs.values())[1].astype(np.float32)
    C_np = np.zeros_like(A_np, dtype=np.float32)

    func(
        A_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        B_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        C_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        M, N
    )

    return C_np
