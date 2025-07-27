import hashlib
from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.utils import topological_sort_operations, get_graph_io

def dim_expr(d):
    return f"{d.name}_val" if isinstance(d, Var) else str(d)

def compute_openblas(outputs):
    if not outputs:
        return "", "", []

    for i, out_node in enumerate(outputs):
        if not getattr(out_node, 'name', None) or out_node.name == "unnamed":
            out_node.name = f"out_{i}"

    ops_sorted = topological_sort_operations(outputs)
    all_inputs, all_dims = get_graph_io(outputs)
    all_matrix_nodes_in_graph = set(all_inputs)
    for op in ops_sorted:
        all_matrix_nodes_in_graph.add(op)
    for out_node in outputs:
        all_matrix_nodes_in_graph.add(out_node)

    add_counter = 0
    sub_counter = 0
    matmul_counter = 0
    temp_op_counter = 0

    for op in ops_sorted:
        if op not in outputs and (op.name is None or op.name == "unnamed"):
            op_type_lower = op.operations_type.lower()
            if op_type_lower == "add":
                op.name = f"intermediate_add_{add_counter}"
                add_counter += 1
            elif op_type_lower == "sub":
                op.name = f"intermediate_sub_{sub_counter}"
                sub_counter += 1
            elif op_type_lower == "matmul":
                op.name = f"intermediate_multiply_{matmul_counter}"
                matmul_counter += 1
            else:
                op.name = f"intermediate_op_{temp_op_counter}"
                temp_op_counter += 1

    sorted_matrix_nodes = sorted(list(all_matrix_nodes_in_graph), key=lambda node: node.name)
    func_args = []
    ordered_arg_names = []
    for node in sorted_matrix_nodes:
        func_args.append(f"float* {node.name}")
        ordered_arg_names.append(node.name)
    sorted_dims = sorted(list(all_dims), key=lambda d: d.name)
    for dim_var in sorted_dims:
        func_args.append(f"int {dim_var.name}")
        ordered_arg_names.append(dim_var.name)
    func_sig = ", ".join(func_args)
    sig_str = str(ordered_arg_names) + str([op.operations_type for op in ops_sorted])
    h = hashlib.md5(sig_str.encode()).hexdigest()[:8]
    c_func_name = f"comp_graph_openblas_{h}"
    code_lines = [
        '#include <cblas.h>\n',
        '#include <cstring>\n',
        '#include <vector>\n',
        f'extern "C" void {c_func_name}({func_sig}) {{\n',
    ]
    for d in sorted_dims:
        code_lines.append(f"    int {d.name}_val = {d.name};\n")
    code_lines.append("\n")
    dsl_to_cpp = {node: node.name for node in all_matrix_nodes_in_graph}
    for op in ops_sorted:
        out = dsl_to_cpp[op]
        a   = dsl_to_cpp[op.inputs[0]]
        b   = dsl_to_cpp[op.inputs[1]] if len(op.inputs) > 1 else None
        m = dim_expr(op.shape[0])
        n = dim_expr(op.shape[1])
        if op.operations_type == "matmul":
            k = dim_expr(op.inputs[0].shape[1])
            code_lines.append(
                f"    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, "
                f"{m}, {n}, {k}, 1.0f, {a}, {k}, {b}, {n}, 0.0f, {out}, {n});\n"
            )
        elif op.operations_type == "add":
            sz = f"(long long){m} * {n}"
            code_lines.append(
                f"    std::memcpy({out}, {a}, {sz} * sizeof(float));\n"
                f"    cblas_saxpy({sz}, 1.0f, {b}, 1, {out}, 1);\n"
            )
        elif op.operations_type == "sub":
            sz = f"(long long){m} * {n}"
            code_lines.append(
                f"    std::memcpy({out}, {a}, {sz} * sizeof(float));\n"
                f"    cblas_saxpy({sz}, -1.0f, {b}, 1, {out}, 1);\n"
            )
        else:
            raise NotImplementedError(f"Unsupported op: {op.operations_type}")
    code_lines.append("}\n")
    return "".join(code_lines), c_func_name, ordered_arg_names



#no op






























