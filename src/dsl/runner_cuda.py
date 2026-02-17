
import subprocess
import numpy as np
import ctypes
import os
import tempfile
import torch  
from dsl.operations import Operation
from dsl.codegen_cuda import generate_cuda_code
from dsl.codegen_numpy import compute_numpy
from dsl.matrix import (
    Matrix, GeneralMatrix, DiagonalMatrix,
    UpperTriangularMatrix, LowerTriangularMatrix, SymmetricMatrix
)
from dsl.var import Var
from dsl.utils import get_graph_io


def get_shape(shape_tuple, inputs):
    real_shape = []
    for dim in shape_tuple:
        if isinstance(dim, Var):
            val = inputs.get(dim.name.lower(), inputs.get(dim.name))
            real_shape.append(int(val))
        elif isinstance(dim, int):
            real_shape.append(dim)
    return tuple(real_shape)

def run(outs, inputs, backend="cuda", out_matrix_obj=None, 
        bm=None, bn=None, bk=None, pipeline=None, swizzle_n=None, num_threads=None):
    
    if not outs: return ({}, 0.0, 0.0)

    actual_out_matrix = out_matrix_obj if out_matrix_obj else outs[0]

    bm_val = int(bm or inputs.get("bm", 128))
    bn_val = int(bn or inputs.get("bn", 128))
    bk_val = int(bk or inputs.get("bk", 16))
    pipe_val = int(pipeline or inputs.get("pipeline", 2))
    swiz_val = int(swizzle_n or inputs.get("swizzle_n", 8))
    nthr_val = int(num_threads or inputs.get("num_threads", 256))

    smem_a_bytes = (bm_val + 4) * bk_val * 4
    smem_b_bytes = (bn_val + 4) * bk_val * 4
    shared_mem_bytes = (smem_a_bytes + smem_b_bytes) * pipe_val

    gen_code, func_name, arg_names = generate_cuda_code(
        outs, actual_out_matrix, 
        bm=bm_val, bn=bn_val, bk=bk_val, 
        pipeline=pipe_val, swizzle_n=swiz_val, 
        num_threads=nthr_val, shared_mem_bytes=shared_mem_bytes
    )

    compiler_flags = [
        "nvcc", "-O3", "-shared", "--compiler-options", "'-fPIC'",
        "-arch=native", "TEMP_CU_PATH", "-o", "TEMP_SO_PATH"
    ]

    lib_file = tempfile.NamedTemporaryFile(suffix=".so", delete=False)
    lib_path = lib_file.name
    lib_file.close()

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=True) as tmp_file:
            tmp_file.write(gen_code)
            tmp_file.flush()
            cmd = [x.replace("TEMP_CU_PATH", tmp_file.name).replace("TEMP_SO_PATH", lib_path) for x in compiler_flags]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print("Compilation Failed:\n", res.stderr)
                raise subprocess.CalledProcessError(res.returncode, cmd, res.stdout, res.stderr)

        lib = ctypes.CDLL(lib_path)
        kernel_func = getattr(lib, func_name)

        all_nodes, _ = get_graph_io(outs)
        obj_map = {n.name: n for n in list(all_nodes) + [actual_out_matrix] if hasattr(n, "name")}

        c_args = []
        out_tensor = None
        
        time_ms = ctypes.c_float(0.0)
        gflops_val = ctypes.c_float(0.0) 

        for name in arg_names:
            clean_name = name.replace("_data", "")
            
            if name == "time_ms_out":
                c_args.append(ctypes.byref(time_ms))
            elif name == "gflops_out": 
                c_args.append(ctypes.byref(gflops_val))
            elif name.endswith("_data"):
                obj = obj_map[clean_name]
                shape = get_shape(obj.shape, inputs)
                
                if isinstance(obj, (SymmetricMatrix, UpperTriangularMatrix, LowerTriangularMatrix)):
                    size = shape[0] * (shape[0] + 1) // 2
                elif isinstance(obj, DiagonalMatrix):
                    size = shape[0]
                else:
                    size = np.prod(shape)
                
                
                if clean_name not in inputs:
                    
                    out_tensor = torch.zeros(size, dtype=torch.float32, device="cuda")
                    d_ptr = out_tensor.data_ptr()
                else:
                    val = inputs[clean_name]
                    if torch.is_tensor(val):
                        
                        if not val.is_cuda: val = val.cuda()
                        if not val.is_contiguous(): val = val.contiguous()
                        val = val.view(-1) 
                        d_ptr = val.data_ptr()
                    
                
                c_args.append(ctypes.c_void_p(d_ptr))
            else:
                val = inputs.get(clean_name, 0)
                c_args.append(ctypes.c_int(int(val)))

        kernel_func(*c_args)

    finally:
        if os.path.exists(lib_path): os.remove(lib_path)

    
    return {actual_out_matrix.name: out_tensor}, time_ms.value, gflops_val.value


def _build_compute_schedule(node, visited, schedule):
    
    if id(node) in visited:
        return
    visited.add(id(node))
    
    
    
    
    if isinstance(node, Operation) and hasattr(node, 'operands'):
        for child in node.operands:
            _build_compute_schedule(child, visited, schedule)
        
        
        schedule.append(node)
def run_graph(outs, inputs, backend="cuda", **kwargs):
    
    ops_to_compute = []
    visited = set()
    for out_node in outs:
        _build_compute_schedule(out_node, visited, ops_to_compute)
        
    if not ops_to_compute:
        ops_to_compute = outs

    current_inputs = inputs.copy()
    total_time_ms = 0.0
    final_outputs = {}

    from dsl.operations import Operation
    from dsl.matrix import GeneralMatrix

    
    for i, op_node in enumerate(ops_to_compute):
        
        original_operands = op_node.operands.copy()
        safe_operands = []
        for child in op_node.operands:
            if isinstance(child, Operation):
                
                safe_operands.append(GeneralMatrix(child.shape, child.name))
            else:
                
                safe_operands.append(child)
        
        op_node.operands = safe_operands
        
        res, t_ms, _ = run(
            outs=[op_node], 
            inputs=current_inputs, 
            backend=backend, 
            out_matrix_obj=op_node, 
            **kwargs
        )
        
        
        op_node.operands = original_operands
        
        
        current_inputs[op_node.name] = res[op_node.name]
        total_time_ms += t_ms
        final_outputs[op_node.name] = res[op_node.name]
        
        
    
    return final_outputs, total_time_ms, 0.0