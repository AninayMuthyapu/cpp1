


import hashlib
from dsl.operations import Operation
from dsl.var import Var
from dsl.matrix import Matrix
from dsl.utils import topological_sort_operations, get_graph_io


def dim_expr(d):
    return f"{d.name}_val" if isinstance(d, Var) else str(d)


def generate_cpp_code(outputs):
    if not outputs:
        return "", "", []

    for i, out_node in enumerate(outputs):
        if not hasattr(out_node, 'name') or out_node.name is None or out_node.name == "unnamed":
            out_node.name = f"out_{i}"

    try:
        ops_sorted = topological_sort_operations(outputs)
    except ValueError as e:
        raise e

    all_inputs, all_dims = get_graph_io(outputs)

    func_args = []
    ordered_arg_names = []

    for mat in all_inputs:
        func_args.append(f"float* {mat.name}")
        ordered_arg_names.append(mat.name)

    for out_node in outputs:
        if out_node not in all_inputs:
            func_args.append(f"float* {out_node.name}")
            ordered_arg_names.append(out_node.name)

    for dim_var in all_dims:
        func_args.append(f"int {dim_var.name}")
        ordered_arg_names.append(dim_var.name)

    func_sig = ", ".join(func_args)

    graph_signature_str = str(ordered_arg_names) + str([op.operations_type for op in ops_sorted])
    unique_hash = hashlib.md5(graph_signature_str.encode()).hexdigest()[:8]
    c_func_name = f"comp_graph_cpp_{unique_hash}"

    code_parts = []
    code_parts.append("""
#include <cstddef>
#include <vector>
#include <cstring>

""")
    code_parts.append(f"extern \"C\" void {c_func_name}({func_sig}) {{\n")
    code_parts.append("    std::vector<float*> temps_to_del;\n\n")

    for dim_var in all_dims:
        code_parts.append(f"    int {dim_var.name}_val = {dim_var.name};\n")
    code_parts.append("\n")

    dsl_to_cpp_map = {}

    add_counter = 0
    sub_counter = 0
    matmul_counter = 0

    for mat in all_inputs:
        dsl_to_cpp_map[mat] = mat.name

    for op in ops_sorted:
        out_cpp_name = None
        is_final_output = op in outputs

        op_type_lower = op.operations_type.lower()

        if is_final_output:
            out_cpp_name = op.name
        else:
            if op_type_lower == "add":
                out_cpp_name = f"add_temp_{add_counter}"
                add_counter += 1
            elif op_type_lower == "sub":
                out_cpp_name = f"sub_temp_{sub_counter}"
                sub_counter += 1
            elif op_type_lower == "matmul":
                out_cpp_name = f"multiply_temp_{matmul_counter}"
                matmul_counter += 1
            else:
                out_cpp_name = f"temp_op_{op_type_lower}_{len(temps_to_del)}"

            m_out = dim_expr(op.shape[0])
            n_out = dim_expr(op.shape[1])

            code_parts.append(f"    float* {out_cpp_name} = new float[(long long){m_out} * {n_out}]();\n")
            code_parts.append(f"    temps_to_del.push_back({out_cpp_name});\n")

        dsl_to_cpp_map[op] = out_cpp_name

        lhs_name = dsl_to_cpp_map.get(op.inputs[0])
        if lhs_name is None:
            raise RuntimeError(f"LHS '{op.inputs[0].name}' not mapped.")

        rhs_name = None
        if len(op.inputs) > 1:
            rhs_name = dsl_to_cpp_map.get(op.inputs[1])
            if rhs_name is None:
                raise RuntimeError(f"RHS '{op.inputs[1].name}' not mapped.")

        m_dim_cpp = dim_expr(op.shape[0])
        n_dim_cpp = dim_expr(op.shape[1])

        if op_type_lower == "add":
            code_parts.append(f"""    for (int i = 0; i < (long long){m_dim_cpp} * {n_dim_cpp}; ++i) {{
        {out_cpp_name}[i] = {lhs_name}[i] + {rhs_name}[i];
    }}
""")
        elif op_type_lower == "sub":
            code_parts.append(f"""    for (int i = 0; i < (long long){m_dim_cpp} * {n_dim_cpp}; ++i) {{
        {out_cpp_name}[i] = {lhs_name}[i] - {rhs_name}[i];
    }}
""")
        elif op_type_lower == "matmul":
            k_dim_cpp = dim_expr(op.inputs[0].shape[1])
            code_parts.append(f"""    for (int i = 0; i < {m_dim_cpp}; ++i) {{
        for (int j = 0; j < {n_dim_cpp}; ++j) {{
            {out_cpp_name}[i*{n_dim_cpp} + j] = 0;
            for (int k = 0; k < {k_dim_cpp}; ++k) {{
                {out_cpp_name}[i*{n_dim_cpp} + j] += {lhs_name}[i*{k_dim_cpp} + k] * {rhs_name}[k*{n_dim_cpp} + j];
            }}
        }}
    }}
""")
        else:
            raise NotImplementedError(f"Op '{op_type_lower}' not supported.")

    code_parts.append("\n    for (float* arr : temps_to_del) {\n")
    code_parts.append("        delete[] arr;\n")
    code_parts.append("    }\n")

    code_parts.append("}\n")

    return "".join(code_parts), c_func_name, ordered_arg_names