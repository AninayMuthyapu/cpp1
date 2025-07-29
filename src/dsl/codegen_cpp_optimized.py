import hashlib
from dsl.operations import Operation
from dsl.var import Var
from dsl.matrix import Matrix
from dsl.utils import topological_sort_operations, get_graph_io 

def dim_expr(d):
    return f"{d.name}_val" if isinstance(d,Var) else str(d)

def compute_optimized(outputs):
    if not outputs:
        return "","",[]
    for i,out_node in enumerate(outputs):
        if not hasattr(out_node,'name') or out_node.name is None or out_node.name =="unnamed":
            out_node.name = f"out_{i}"

    try:
        ops_sorted=topological_sort_operations(outputs)
    except ValueError as e:
        raise e
    
    all_inputs,all_dims=get_graph_io(outputs)
    all_matrix_nodes_in_graph=set(all_inputs)
    for op in ops_sorted:
        all_matrix_nodes_in_graph.add(op)
    for out_node in outputs:
        all_matrix_nodes_in_graph.add(out_node)

    add_counter=0
    sub_counter=0
    temp_op_counter=0   

    for op in ops_sorted:
        if op not in outputs and (op.name is None or op.name == "unnamed"):
            op_type_lower = op.operations_type.lower()
            if op_type_lower == "add":
                op.name = f"intermediate_add_{add_counter}"
                add_counter += 1
            elif op_type_lower == "sub":
                op.name = f"intermediate_sub_{sub_counter}"
                sub_counter += 1
            else:
                op.name = f"intermediate_op_{temp_op_counter}"
                temp_op_counter += 1

    sorted_matrix_nodes = sorted(list(all_matrix_nodes_in_graph), key=lambda node: node.name)   
    func_args=[]
    ordered_arg_names=[]

    for node in sorted_matrix_nodes:
        func_args.append(f"float* {node.name}")
        ordered_arg_names.append(node.name)

    sorted_dims=sorted(list(all_dims), key=lambda d: d.name if isinstance(d, Var) else str(d))

    for dim_var in sorted_dims:
        func_args.append(f"int {dim_expr(dim_var)}")    
        ordered_arg_names.append(dim_expr(dim_var)) 

    func_sig = ", ".join(func_args)
    graph_signature_str = str(ordered_arg_names) + str([op.operations_type for op in ops_sorted])
    unique_hash = hashlib.md5(graph_signature_str.encode()).hexdigest()[:8]     
    c_func_name = f"comp_graph_cpp_{unique_hash}"

    code_parts=[
        '#include <cstddef>\n',
        '#include <cstring>\n',
        '#include <vector>\n',
        '#include <iostream>\n',
        '#include <omp.h>\n',
    ]

    code_parts.append(f"""

void add_op(float* out, const float* a, const float* b, long long size) {{
    #pragma omp parallel for    
    long long avx_size=size / 8 * 8;
    for (long long i = 0; i < size; i += avx_size) {{
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        __m256 out_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(out + i, out_vec);
    }}
     for (long long i =avx_size;i<size;i++){{
        out[i] = a[i] + b[i];
     }}  

void sub_op(float* out, const float* a, const float* b, long long size) {{
    long long avx_size = (size / 8) * 8;

    #pragma omp parallel for
    for (long long i = 0; i < avx_size; i += 8) {{
        __m256 a_vec = _mm256_loadu_ps(&a[i]);
        __m256 b_vec = _mm256_loadu_ps(&b[i]);
        __m256 c_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(&out[i], c_vec);
    }}

    for (long long i = avx_size; i < size; ++i) {{
        out[i] = a[i] - b[i];
    }}
}})""")
    
    code_parts.append(f"extern \"C\" void {c_func_name}({func_sig}) {{\n")

    for dim_var in sorted_dims:
        code_parts.append(f"  int {dim_var.name}_val={dim_var.name};\n")

    code_parts.append("\n")
    dsl_to_cpp_map = {node: node.name for node in all_matrix_nodes_in_graph}

    for op in ops_sorted:
        out_cpp_name=dsl_to_cpp_map[op]

        lhs_name = dsl_to_cpp_map.get(op.inputs[0])
        if lhs_name is None:
            raise RuntimeError(f"LHS '{op.inputs[0].name}' not mapped.")
        
        rhs_name =None
        if len(op.inputs)>1:
            rhs_name= dsl_to_cpp_map.get(op.inputs[1])
            if rhs_name is None:
                raise RuntimeError(f"RHS '{op.inputs[1].name}' not mapped.")
            
        m_dim_cpp=dim_expr(op.shape[0])
        n_dim_cpp=dim_expr(op.shape[1])
        if op_type_lower=="add":
            size_expr=f"(long long ){m_dim_cpp}*{n_dim_cpp}"
            code_parts.append(f"    add_op({out_cpp_name}, {lhs_name}, {rhs_name}, {size_expr});\n")

        if op_type_lower=="sub":
            size_expr = f"(long long){m_dim_cpp} * {n_dim_cpp}"
            code_parts.append(f"    sub_op({out_cpp_name}, {lhs_name}, {rhs_name}, {size_expr});\n")\
        
        else:
            raise NotImplementedError(f"Op '{op_type_lower}'not supported")
        
    code_parts.append("}\n")
    return "".join(code_parts),c_func_name,ordered_arg_names



