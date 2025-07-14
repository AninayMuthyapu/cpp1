import subprocess
import numpy as np
import ctypes
import os
from dsl.codegen import generate_cpp_code
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
    cpp_code = generate_cpp_code(outputs)
    cpp_path = "generated.cpp"
    so_path = "generated.so"

    with open(cpp_path, "w") as f:
        f.write(cpp_code)

    compile_cmd = [
        "g++", "-O3", "-std=c++17", "-fPIC", "-shared",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        cpp_path, "-o", so_path,
        "-L", os.environ["CONDA_PREFIX"] + "/lib", "-lopenblas"
    ]
    print("Compiling with:", " ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)

    lib = ctypes.CDLL(os.path.abspath(so_path))

    output_var = outputs[0]
    output_name = getattr(output_var, "name", None) or "C"
    output_var.name = output_name
    M, N = extract_shape(output_var.shape, inputs)

    C = np.zeros((M, N), dtype=np.float32)

    ptrs = {}
    for name, array in inputs.items():
        if isinstance(array, np.ndarray):
            arr32 = array.astype(np.float32)
            ptrs[name] = arr32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    C_ptr = C.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    lib.compute(ptrs["A"], ptrs["B"], C_ptr, M, N)

    return {output_name: C}



