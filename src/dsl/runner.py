import subprocess
import numpy as np
import ctypes
import os
from dsl.codegen_openblas import compute_openblas
from dsl.codegen import generate_cpp_code
from dsl.codegen_numpy import compute_numpy
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

def run(outputs, inputs, backend="openblas"):
    output_var = outputs[0]
    output_name = getattr(output_var, "name", None) or "C"
    output_var.name = output_name
    M, N = extract_shape(output_var.shape, inputs)

    if backend == "numpy":
        func = compute_numpy([output_var])
        return {output_name: func(inputs)}
    
    elif backend == "cpp":
        cpp_code = generate_cpp_code([output_var])
        cpp_path = "generated_cpp.cpp"
        so_path = "generated_cpp.so"
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        compile_cmd = [
            "/usr/bin/g++",
            "-O3", "-std=c++17", "-fPIC", "-shared",
            cpp_path, "-o", so_path
        ]


        print("Compiling C++ with:", " ".join(compile_cmd))
        subprocess.run(compile_cmd, check=True)
        lib = ctypes.CDLL(os.path.abspath(so_path))
    
    elif backend == "openblas":
        cpp_code = compute_openblas([output_var])
        cpp_path = "generated_openblas.cpp"
        so_path = "generated_openblas.so"
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        compile_cmd = [
            "/usr/bin/g++",
            "-O3", "-std=c++17", "-fPIC", "-shared",
            
            "-I", "/usr/include/x86_64-linux-gnu",
            cpp_path, "-o", so_path,
            "-L", "/usr/lib/x86_64-linux-gnu", "-lopenblas"
        ]
        print("Compiling OpenBLAS with:", " ".join(compile_cmd))
        subprocess.run(compile_cmd, check=True)
        lib = ctypes.CDLL(os.path.abspath(so_path))

    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Prepare input and output buffers
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



