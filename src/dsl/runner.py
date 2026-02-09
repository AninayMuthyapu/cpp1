


import subprocess
import numpy as np
import ctypes
import os
import tempfile
import multiprocessing
from dsl.codegen_special import generate_special_cpp
from dsl.codegen_numpy import compute_numpy
from dsl.matrix import (
    Matrix, GeneralMatrix, DiagonalMatrix,
    UpperTriangularMatrix, LowerTriangularMatrix, SymmetricMatrix
)
from dsl.var import Var
from dsl.utils import get_graph_io, topological_sort_operations

_GLOBAL_CACHES = {}

_libc = ctypes.CDLL("libc.so.6")
_libc.aligned_alloc.restype = ctypes.c_void_p
_libc.aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_libc.free.argtypes = [ctypes.c_void_p]

def _aligned_alloc_float(num_floats, page_size):
    bytes_needed = int(num_floats) * np.dtype(np.float32).itemsize
    alloc_size = ((bytes_needed + page_size - 1) // page_size) * page_size
    ptr = _libc.aligned_alloc(page_size, alloc_size)
    if not ptr:
        raise MemoryError("aligned_alloc failed")
    float_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
    ctypes.memset(float_ptr, 0, alloc_size)
    return float_ptr

def allocate_persistent_caches(bm, bn, bk, max_threads):
    key = (int(bm), int(bn), int(bk), int(max_threads))
    if key in _GLOBAL_CACHES:
        return _GLOBAL_CACHES[key]

    page_size = os.sysconf("SC_PAGESIZE")
    A_cache, B_cache, C_cache = [], [], []

    for _ in range(max_threads):
        A_cache.append(_aligned_alloc_float(bm * bk, page_size))
        B_cache.append(_aligned_alloc_float(bk * bn, page_size))
        C_cache.append(_aligned_alloc_float(bm * bn, page_size))

    cfg = {"A": A_cache, "B": B_cache, "C": C_cache}
    _GLOBAL_CACHES[key] = cfg
    return cfg

def get_shape(shape_tuple, inputs):
    real_shape = []
    for dim in shape_tuple:
        if isinstance(dim, Var):
            val = inputs.get(dim.name.lower(), inputs.get(dim.name))
            if val is None:
                raise ValueError(f"Dimension '{dim.name}' missing")
            real_shape.append(int(val))
        elif isinstance(dim, int):
            real_shape.append(dim)
        else:
            raise TypeError(f"Unsupported dimension type: {type(dim)}")
    return tuple(real_shape)

def expand_matrix_output(obj, arr, inputs):
    shape = get_shape(obj.shape, inputs)
    if isinstance(obj, SymmetricMatrix):
        n = shape[0]
        full = np.zeros((n, n), dtype=np.float32)
        iu = np.triu_indices(n)
        full[iu] = arr
        full[(iu[1], iu[0])] = arr
        return full
    elif isinstance(obj, UpperTriangularMatrix):
        n = shape[0]
        full = np.zeros((n, n), dtype=np.float32)
        iu = np.triu_indices(n)
        full[iu] = arr
        return full
    elif isinstance(obj, LowerTriangularMatrix):
        n = shape[0]
        full = np.zeros((n, n), dtype=np.float32)
        il = np.tril_indices(n)
        full[il] = arr
        return full
    elif isinstance(obj, DiagonalMatrix):
        n = shape[0]
        return np.diag(arr)
    else:
        return arr.reshape(shape)

def _make_ptr_array_for_kernel(ptr_list, max_threads):
    ArrayType = ctypes.POINTER(ctypes.c_float) * max_threads
    if isinstance(ptr_list, list):
        if len(ptr_list) == max_threads:
            arr = ArrayType(*ptr_list)
        elif len(ptr_list) == 1:
            arr = ArrayType(*([ptr_list[0]] * max_threads))
        else:
            arr_list = (ptr_list + [ptr_list[-1]] * max_threads)[:max_threads]
            arr = ArrayType(*arr_list)
    else:
        arr = ArrayType(*([ptr_list] * max_threads))
    return ctypes.cast(arr, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)))

def run(outs, inputs, backend="special", out_matrix_obj=None,
        bm=None, bn=None, bk=None, it_m=None, it_n=None, it_k=None):

    if not outs: return ({}, 0.0, 0.0)
    if backend == "numpy":
        return compute_numpy(outs)(inputs), 0.0, 0.0

    actual_out_matrix = out_matrix_obj if out_matrix_obj else outs[0]
    DEFAULT_BM, DEFAULT_BN, DEFAULT_BK = 64, 64, 64
    DEFAULT_IT_M, DEFAULT_IT_N, DEFAULT_IT_K = 4, 32, 4
    max_threads = multiprocessing.cpu_count()

    if backend == "special":
        gen_cpp_code, func_name, arg_names_ordered = generate_special_cpp(outs, actual_out_matrix)
        compiler_flags = ["g++","-O3","-std=c++17","-fPIC","-shared","-fopenmp",
                          "TEMP_CPP_PATH","-o","TEMP_SO_PATH"]
    elif backend == "cpp_avx":
        from dsl.codegen_multiply_avx import compute_optimized as compute_matmul_cpp
        op = getattr(outs[0], "operator", getattr(outs[0], "op", None))
        if op == "add" or op=="sub":
            from dsl.codegen_cpp_optimized import compute_optimized
            gen_cpp_code, func_name, arg_names_ordered = compute_optimized(outs, actual_out_matrix)
            compiler_flags = ["g++","-O3","-std=c++17","-fPIC","-shared","-fopenmp",
                          "-march=native","-mavx2","-mfma",
                          "TEMP_CPP_PATH","-o","TEMP_SO_PATH"]
        if op=="matmul" or op=="mul":
            bm = int(bm or inputs.get("bm", DEFAULT_BM))
            bn = int(bn or inputs.get("bn", DEFAULT_BN))
            bk = int(bk or inputs.get("bk", DEFAULT_BK))
            it_m = int(it_m or inputs.get("it_m", DEFAULT_IT_M))
            it_n = int(it_n or inputs.get("it_n", DEFAULT_IT_N))
            it_k = int(it_k or inputs.get("it_k", DEFAULT_IT_K))

            cache_config = allocate_persistent_caches(bm, bn, bk, max_threads)

            gen_cpp_code, func_name, arg_names_ordered = compute_matmul_cpp(
                outs, actual_out_matrix
                )

            compiler_flags = ["g++","-O3","-std=c++17","-fPIC","-shared","-fopenmp",
                          "-march=native","-mavx2","-mfma",
                          "TEMP_CPP_PATH","-o","TEMP_SO_PATH"]

    elif backend == "openblas":
        from dsl.codegen_openblas import compute_openblas
        gen_cpp_code, func_name, arg_names_ordered = compute_openblas(outs, actual_out_matrix)
        compiler_flags = ["g++","-O3","-std=c++17","-fPIC","-shared","-fopenmp","-lopenblas",
                          "TEMP_CPP_PATH","-o","TEMP_SO_PATH"]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    lib_file = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
    lib_path = lib_file.name
    lib_file.close()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=True) as tmp_cpp_file:
        tmp_cpp_file.write(gen_cpp_code)
        tmp_cpp_file.flush()
        compile_cmd = [arg.replace("TEMP_CPP_PATH", tmp_cpp_file.name).replace("TEMP_SO_PATH", lib_path)
                       for arg in compiler_flags]
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Compilation failed.", result.stdout, result.stderr)
            raise subprocess.CalledProcessError(result.returncode, compile_cmd, result.stdout, result.stderr)

    lib = ctypes.CDLL(lib_path)
    c_func = getattr(lib, func_name)

    time_from_c = ctypes.c_double(0)
    gflops_from_c = ctypes.c_double(0)
    os.environ.setdefault("OMP_NUM_THREADS", str(min(max_threads, 8)))

    all_nodes, _ = get_graph_io(outs)
    obj_map = {n.name: n for n in list(all_nodes) + [actual_out_matrix] if hasattr(n, "name")}
    c_args, np_refs = [], []

    for arg_name in arg_names_ordered:
        base_name = arg_name.replace("_data", "")
        if arg_name == "time_ms_out":
            c_args.append(ctypes.byref(time_from_c))
        elif arg_name == "gflops_out":
            c_args.append(ctypes.byref(gflops_from_c))
        elif arg_name.endswith("_data"):
            sym_obj = obj_map[base_name]
            if base_name not in inputs:
                shape = get_shape(sym_obj.shape, inputs)
                if isinstance(sym_obj, (SymmetricMatrix, UpperTriangularMatrix, LowerTriangularMatrix)):
                    np_arr = np.zeros(shape[0] * (shape[0] + 1) // 2, dtype=np.float32)
                elif isinstance(sym_obj, DiagonalMatrix):
                    np_arr = np.zeros(shape[0], dtype=np.float32)
                else:
                    np_arr = np.zeros(shape, dtype=np.float32)
                inputs[base_name] = np_arr
            else:
                np_arr = np.ascontiguousarray(inputs[base_name].astype(np.float32))
                inputs[base_name] = np_arr
            np_refs.append(np_arr)
            c_args.append(np_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        elif arg_name in ("A_cache", "B_cache", "C_cache"):
            key = arg_name[0]
            ptr_list = cache_config[key]
            float_pp = _make_ptr_array_for_kernel(ptr_list, max_threads)
            c_args.append(float_pp)
        else:
            if base_name == "max_threads":
                c_args.append(ctypes.c_int(max_threads))
            elif base_name in ["bm","bn","bk","it_m","it_n","it_k"]:
                continue
            else:
                c_args.append(ctypes.c_int(int(inputs.get(base_name, 0))))

    try:
        c_func(*c_args)
    except Exception as e:
        print("ERROR: Kernel call raised an exception:", e)
        raise

    expanded = expand_matrix_output(actual_out_matrix, inputs[actual_out_matrix.name], inputs)
    return {actual_out_matrix.name: expanded}, time_from_c.value, gflops_from_c.value









