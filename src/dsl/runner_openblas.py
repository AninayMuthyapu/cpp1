import subprocess
import numpy as np
import ctypes
import os
from dsl.codegen_openblas import compute_openblas
from dsl.var import Var

def extract_shape(shape, inputs):
    resolved = []
    for dim in shape:
        if isinstance(dim, Var):
            resolved.append(inputs[dim.name])
        elif isinstance(dim, int):
            resolved.append(dim)
        else:
            raise ValueError(f"Invalid dimension: {dim}")
    return tuple(resolved)

def run(outputs, inputs):
    cpp_code = compute_openblas([outputs[0]])
    cpp_path = "generated_openblas.cpp"
    so_path = "generated_openblas.so"
    with open(cpp_path, "w") as f:
        f.write(cpp_code)
    compile_cmd = [
        "/usr/bin/g++",
        "-O3", "-std=c++17", "-fPIC", "-shared",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-I", "/usr/include/x86_64-linux-gnu",
        cpp_path, "-o", so_path,
        "-L", "/usr/lib/x86_64-linux-gnu", "-lopenblas"
    ]
    print("Compiling OpenBLAS with:", " ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)
    lib = ctypes.CDLL(os.path.abspath(so_path))
    output_var = outputs[0]
    output_name = getattr(output_var, "name", None) or "C"
    output_var.name = output_name
    M, N = extract_shape(output_var.shape, inputs)
    K = inputs["K"]
    C = np.zeros((M, N), dtype=np.float32)
    ptrs = {}
    for name, array in inputs.items():
        if isinstance(array, np.ndarray):
            arr32 = array.astype(np.float32)
            ptrs[name] = arr32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    K = extract_shape(inputs["A"].shape, inputs)[1]
    lib.compute(ptrs["A"], ptrs["B"], C_ptr, M, N, K)
    return {output_name: C}
