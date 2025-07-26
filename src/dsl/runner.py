import subprocess
import numpy as np
import ctypes
import os
from dsl.codegen_openblas import compute_openblas
from dsl.codegen import generate_cpp_code
from dsl.codegen_numpy import compute_numpy
from dsl.var import Var
from dsl.matrix import Matrix 
from dsl.utils import get_graph_io, topological_sort_operations
from dsl.operations import Operation

def get_shape(shape_tuple, in_data):
    res = []
    for dim in shape_tuple:
        if isinstance(dim, Var):
            if dim.name not in in_data:
                raise ValueError(f"Dim '{dim.name}' not in inputs.")
            res.append(in_data[dim.name])
        elif isinstance(dim, int):
            res.append(dim)
        else:
            raise TypeError(f"Invalid dim type: {type(dim)}. Expected Var or int.")
    return tuple(res)

def run(outs, inputs, backend="openblas"):
    if not outs:
        return {}

    if backend == "numpy":
        np_eval_func = compute_numpy(outs)
        return np_eval_func(inputs)

    elif backend in ["cpp", "openblas"]:
        cpp_code = ""
        comp_cmd = []
        c_func_name = "" 
        ordered_c_func_arg_names = []

        if backend == "cpp":
            cpp_code, c_func_name, ordered_c_func_arg_names = generate_cpp_code(outs)
        elif backend == "openblas":
            cpp_code, c_func_name, ordered_c_func_arg_names = compute_openblas(outs)
        
        cpp_path = f"gen_{c_func_name}.cpp"
        so_path = f"gen_{c_func_name}.so"

        if backend == "cpp":
            comp_cmd = [
                "/usr/bin/g++",
                "-O3", "-std=c++17", "-fPIC", "-shared",
                cpp_path, "-o", so_path
            ]
        elif backend == "openblas":
            comp_cmd = [
                "/usr/bin/g++",
                "-O3", "-std=c++17", "-fPIC", "-shared",
                "-I", "/usr/include/x86_64-linux-gnu",
                cpp_path, "-o", so_path,
                "-L", "/usr/lib/x86_64-linux-gnu",
                "-lopenblas"
            ]

        with open(cpp_path, "w") as f:
            f.write(cpp_code)

        try:
            subprocess.run(comp_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise

        lib = ctypes.CDLL(os.path.abspath(so_path))
        c_func = getattr(lib, c_func_name)

        concrete_data_map = {}
        np_arr_refs = []

        all_graph_nodes = set()
        all_inputs, all_sym_dims = get_graph_io(outs)
        for node in all_inputs:
            all_graph_nodes.add(node)
        for op in topological_sort_operations(outs):
            all_graph_nodes.add(op)
        for out_node in outs:
            all_graph_nodes.add(out_node)
        
        for node in all_graph_nodes:
            if isinstance(node, (Matrix, Operation)):
                if node.name not in inputs:
                    m_val, n_val = get_shape(node.shape, inputs)
                    arr = np.zeros((m_val, n_val), dtype=np.float32)
                    concrete_data_map[node.name] = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    np_arr_refs.append(arr)
                else:
                    arr = inputs[node.name]
                    arr_f32 = arr.astype(np.float32, copy=False)
                    concrete_data_map[node.name] = arr_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    np_arr_refs.append(arr_f32)

        for dim_var in all_sym_dims:
            if dim_var.name not in inputs:
                raise ValueError(f"Value for dim '{dim_var.name}' not in inputs.")
            concrete_data_map[dim_var.name] = inputs[dim_var.name] 

        c_arg_types = []
        c_arg_vals = []

        for arg_name in ordered_c_func_arg_names:
            if arg_name in concrete_data_map:
                if isinstance(concrete_data_map[arg_name], (ctypes.POINTER(ctypes.c_float), ctypes._Pointer)):
                    c_arg_types.append(ctypes.POINTER(ctypes.c_float))
                    c_arg_vals.append(concrete_data_map[arg_name])
                elif isinstance(concrete_data_map[arg_name], int):
                    c_arg_types.append(ctypes.c_int)
                    c_arg_vals.append(ctypes.c_int(concrete_data_map[arg_name]))
                else:
                    raise TypeError(f"Unexpected type for concrete data map entry '{arg_name}': {type(concrete_data_map[arg_name])}")
            else:
                raise RuntimeError(f"input functions are not prealocated ")
                                   

        c_func.argtypes = c_arg_types
        c_func.restype = None

        c_func(*c_arg_vals)

        out_res_dict = {}
        for out_node in outs:
            if out_node.name not in concrete_data_map:
                raise RuntimeError(f"Output '{out_node.name}' not found in concrete_data_map after C++ execution.")
            m_out, n_out = get_shape(out_node.shape, inputs)
            out_res_dict[out_node.name] = np.ctypeslib.as_array(concrete_data_map[out_node.name], shape=(m_out, n_out))

        os.remove(cpp_path)
        os.remove(so_path)

        return out_res_dict

    else:
        raise ValueError(f"wrong backend")
































# import subprocess
# import numpy as np
# import ctypes
# import os
# from dsl.codegen_openblas import compute_openblas
# from dsl.codegen import generate_cpp_code 
# from dsl.codegen_numpy import compute_numpy
# from dsl.var import Var
# from dsl.matrix import Matrix 
# from dsl.utils import get_graph_io

# def get_shape(shape_tuple, in_data):
#     res = []
#     for dim in shape_tuple:
#         if isinstance(dim, Var):
#             if dim.name not in in_data:
#                 raise ValueError(f"Dim '{dim.name}' not in inputs.")
#             res.append(in_data[dim.name])
#         elif isinstance(dim, int):
#             res.append(dim)
#         else:
#             raise TypeError(f"Invalid dim type: {type(dim)}. Expected Var or int.")
#     return tuple(res)

# def run(outs, inputs, backend="openblas"):
#     if not outs:
#         return {}

#     if backend == "numpy":
#         np_eval_func = compute_numpy(outs)
#         return np_eval_func(inputs)

#     elif backend in ["cpp", "openblas"]:
#         cpp_code = ""
#         comp_cmd = []
#         c_func_name = "" 
#         ordered_c_func_arg_names = []

#         if backend == "cpp":
#             cpp_code, c_func_name, ordered_c_func_arg_names = generate_cpp_code(outs)
#         elif backend == "openblas":
#             cpp_code, c_func_name, ordered_c_func_arg_names = compute_openblas(outs)
        
#         cpp_path = f"gen_{c_func_name}.cpp"
#         so_path = f"gen_{c_func_name}.so"

#         if backend == "cpp":
#             comp_cmd = [
#                 "/usr/bin/g++",
#                 "-O3", "-std=c++17", "-fPIC", "-shared",
#                 cpp_path, "-o", so_path
#             ]
#         elif backend == "openblas":
#             comp_cmd = [
#                 "/usr/bin/g++",
#                 "-O3", "-std=c++17", "-fPIC", "-shared",
#                 "-I", "/usr/include/x86_64-linux-gnu",
#                 cpp_path, "-o", so_path,
#                 "-L", "/usr/lib/x86_64-linux-gnu",
#                 "-lopenblas"
#             ]

#         with open(cpp_path, "w") as f:
#             f.write(cpp_code)

#         try:
#             subprocess.run(comp_cmd, check=True, capture_output=True, text=True)
#         except subprocess.CalledProcessError as e:
#             print(f"Compilation failed. STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
#             raise

#         lib = ctypes.CDLL(os.path.abspath(so_path))
#         c_func = getattr(lib, c_func_name)

#         concrete_data_map = {}
#         np_arr_refs = []

#         all_in_mats, all_sym_dims = get_graph_io(outs) 

#         for mat in all_in_mats:
#             if mat.name not in inputs:
#                raise ValueError(f"Data for '{mat.name}' not in inputs.")
#             np_arr = inputs[mat.name]
#             np_arr_f32 = np_arr.astype(np.float32, copy=False) 
#             concrete_data_map[mat.name] = np_arr_f32.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#             np_arr_refs.append(np_arr_f32)

#         out_res_dict = {}
#         for out_node in outs:
#             if out_node not in all_in_mats:
#                 m_out, n_out = get_shape(out_node.shape, inputs)
#                 out_np_arr = np.zeros((m_out, n_out), dtype=np.float32)
#                 concrete_data_map[out_node.name] = out_np_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
#                 np_arr_refs.append(out_np_arr)
#                 out_res_dict[out_node.name] = out_np_arr
#             else:
#                 out_res_dict[out_node.name] = inputs[out_node.name]

#         for dim_var in all_sym_dims:
#             if dim_var.name not in inputs:
#                 raise ValueError(f"Value for dim '{dim_var.name}' not in inputs.")
#             concrete_data_map[dim_var.name] = inputs[dim_var.name] 

#         c_arg_types = []
#         c_arg_vals = []

#         for arg_name in ordered_c_func_arg_names:
#             if arg_name in concrete_data_map:
#                 if isinstance(concrete_data_map[arg_name], (ctypes.POINTER(ctypes.c_float), ctypes._Pointer)):
#                     c_arg_types.append(ctypes.POINTER(ctypes.c_float))
#                     c_arg_vals.append(concrete_data_map[arg_name])
#                 elif isinstance(concrete_data_map[arg_name], int):
#                     c_arg_types.append(ctypes.c_int)
#                     c_arg_vals.append(ctypes.c_int(concrete_data_map[arg_name]))
#                 else:
#                     raise TypeError(f"Unexpected type for concrete data map entry '{arg_name}': {type(concrete_data_map[arg_name])}")
#             else:
#                 raise RuntimeError(f"Argument '{arg_name}' from C++ signature not found in concrete data map.")

#         c_func.argtypes = c_arg_types
#         c_func.restype = None

#         c_func(*c_arg_vals)

#         os.remove(cpp_path)
#         os.remove(so_path)

#         return out_res_dict

#     else:
#         raise ValueError(f"Bad backend: {backend}")