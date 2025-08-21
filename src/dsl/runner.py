import subprocess
import numpy as np
import ctypes
import os
import tempfile

from dsl.codegen_openblas import compute_openblas
from dsl.codegen import generate_cpp_code
from dsl.codegen_numpy import compute_numpy
from dsl.codegen_cpp_optimized import compute_optimized
from dsl.codegen_special import generate_special_cpp

from dsl.matrix import Matrix, DiagonalMatrix, UpperTriangularMatrix, LowerTriangularMatrix, ToeplitzMatrix
from dsl.operations import Operation
from dsl.var import Var
from dsl.utils import get_graph_io, topological_sort_operations

def get_shape(shape_tuple, in_data):
    real_shape = []
    for dim in shape_tuple:
        if isinstance(dim, Var):
            if dim.name not in in_data:
                raise ValueError(f"Dimension '{dim.name}' missing. Add it to inputs.")
            real_shape.append(in_data[dim.name])
        elif isinstance(dim, int):
            real_shape.append(dim)
        else:
            raise TypeError(f" Needs Var or int.")
    return tuple(real_shape)

def run(outs, inputs, backend="openblas", out_matrix_obj=None): 
    if not outs:
        return {}

    if backend == "numpy":
        np_func = compute_numpy(outs)
        return np_func(inputs)

    gen_cpp_code = ""
    func_name = ""
    arg_names_ordered = []
    compiler_flags = []
    backend_label = ""

    actual_out_matrix = out_matrix_obj if out_matrix_obj else outs[0]

    if backend == "cpp_avx":
        gen_cpp_code, func_name, arg_names_ordered = compute_optimized(outs, actual_out_matrix) 
        backend_label = "cpp_avx"
        compiler_flags = [
            "g++", "-O3", "-std=c++17", "-fPIC", "-shared", "-fopenmp", "-march=native",
            "TEMP_CPP_PATH", "-o", "TEMP_SO_PATH"
        ]
    elif backend == "cpp":
        gen_cpp_code, func_name, arg_names_ordered = generate_cpp_code(outs, actual_out_matrix) 
        backend_label = "cpp"
        compiler_flags = [
            "g++", "-O3", "-std=c++17", "-fPIC", "-shared",
            "TEMP_CPP_PATH", "-o", "TEMP_SO_PATH"
        ]
    elif backend == "openblas":
        gen_cpp_code, func_name, arg_names_ordered = compute_openblas(outs, actual_out_matrix) 
        backend_label = "openblas"
        compiler_flags = [
            "g++", "-O3", "-std=c++17", "-fPIC", "-shared",
            "-I", "/usr/include/x86_64-linux-gnu",
            "TEMP_CPP_PATH", "-o", "TEMP_SO_PATH",
            "-L", "/usr/lib/x86_64-linux-gnu",
            "-lopenblas"
        ]
    elif backend == "special":
        gen_cpp_code, func_name, arg_names_ordered = generate_special_cpp(outs, actual_out_matrix) 
        backend_label = "special"
        compiler_flags = [
            "g++", "-O3", "-std=c++17", "-fPIC", "-shared",
            "TEMP_CPP_PATH", "-o", "TEMP_SO_PATH"
        ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".cpp", delete=False) as tmp_cpp_file:
        tmp_cpp_file.write(gen_cpp_code)
        cpp_src_path = tmp_cpp_file.name

    lib_path = tempfile.mktemp(suffix=".so")

    final_comp_cmd = [
        arg.replace("TEMP_CPP_PATH", cpp_src_path).replace("TEMP_SO_PATH", lib_path)
        for arg in compiler_flags
    ]

    print(f"\n--- Generated C++ Code \n")
    print(gen_cpp_code)
    print(f"\n--- End Generated C++ Code \n")

    try:
        subprocess.run(final_comp_cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        print(f"Compilation failed ")
        os.remove(cpp_src_path)
        if os.path.exists(lib_path):
            os.remove(lib_path)
        raise

    compiled_lib = ctypes.CDLL(os.path.abspath(lib_path))
    c_func = getattr(compiled_lib, func_name)

    c_arg_types = []
    c_arg_values = []
    np_array_refs = []

    all_graph_elements = set()
    all_inputs_from_graph, _ = get_graph_io(outs)

    for item in all_inputs_from_graph:
        all_graph_elements.add(item)
    all_graph_elements.add(actual_out_matrix) 

    for op_node in topological_sort_operations(outs):
        all_graph_elements.add(op_node)
    
    obj_by_name = {obj.name: obj for obj in all_graph_elements if isinstance(obj, (Matrix, Operation))}

    for arg_name in arg_names_ordered:
        if arg_name.endswith('_data'):
            mat_name = arg_name.replace('_data', '')
            sym_obj = None
            if mat_name in inputs:
                np_arr = inputs[mat_name].astype(np.float32, copy=False)
                sym_obj = next((m for m in all_graph_elements if m.name == mat_name and isinstance(m, Matrix)), None)
            else:
                sym_obj = obj_by_name.get(mat_name)
                if mat_name == actual_out_matrix.name:
                    sym_obj = actual_out_matrix
            
            concrete_logical_shape = get_shape(sym_obj.shape, inputs)
            N_val = inputs['N']


            if isinstance(sym_obj, (UpperTriangularMatrix, LowerTriangularMatrix)):
                num_elements = N_val * (N_val + 1) // 2
                actual_storage_shape = (num_elements,)

            elif isinstance(sym_obj, DiagonalMatrix):
                actual_storage_shape = (N_val,)
            elif isinstance(sym_obj, ToeplitzMatrix):
                num_elements = (2 * N_val) - 1
                actual_storage_shape = (num_elements,)
            else:
                actual_storage_shape = concrete_logical_shape
            if mat_name in inputs:
                np_arr = inputs[mat_name].astype(np.float32, copy=False)
                if np_arr.shape != actual_storage_shape:
                    raise ValueError(f"Error in shape")
            else:
                np_arr = np.zeros(actual_storage_shape, dtype=np.float32)
            inputs[mat_name] = np_arr
            c_arg_val = inputs[mat_name].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            c_arg_types.append(ctypes.POINTER(ctypes.c_float))
            c_arg_values.append(c_arg_val)
            np_array_refs.append(inputs[mat_name])
        elif arg_name in inputs and isinstance(inputs[arg_name], (int, np.integer)):
            c_arg_val = ctypes.c_int(inputs[arg_name])
            c_arg_types.append(ctypes.c_int)
            c_arg_values.append(c_arg_val)
        
        
    c_func.argtypes = c_arg_types
    c_func.restype = None
    c_func(*c_arg_values)

    results_dict = {}
    
    results_dict[actual_out_matrix.name] = inputs[actual_out_matrix.name]

    os.remove(cpp_src_path)
    os.remove(lib_path) 

    return results_dict
