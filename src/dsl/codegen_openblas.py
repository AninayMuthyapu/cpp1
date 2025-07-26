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



































# import hashlib
# from dsl.utils import topological_sort_operations, get_graph_io
# from dsl.var import Var
# from dsl.operations import Operation


# def dim_expr(d):
#     return f"{d.name}_val" if isinstance(d, Var) else str(d)


# def compute_openblas(outputs):
#     if not outputs:
#         return "", "", []

    
#     for i, out in enumerate(outputs):
#         if not getattr(out, 'name', None) or out.name == "unnamed":
#             out.name = f"out_{i}"

#     ops_sorted = topological_sort_operations(outputs)
#     all_inputs, all_dims = get_graph_io(outputs)
    
#     ops_sorted = topological_sort_operations(outputs)
#     print("ops_sorted =", ops_sorted)          

#     all_inputs = sorted(all_inputs, key=lambda t: t.name)
#     all_dims   = sorted(all_dims,   key=lambda d: d.name)

#     func_args, ordered_arg_names = [], []

    
#     for mat in all_inputs:
#         func_args.append(f"float* {mat.name}")
#         ordered_arg_names.append(mat.name)

    
#     for out in outputs:
#         if out not in all_inputs:
#             func_args.append(f"float* {out.name}")
#             ordered_arg_names.append(out.name)

    
#     for d in all_dims:
#         func_args.append(f"int {d.name}")
#         ordered_arg_names.append(d.name)

#     func_sig = ", ".join(func_args)

#     sig_str = str(ordered_arg_names) + str([op.operations_type for op in ops_sorted])
#     h = hashlib.md5(sig_str.encode()).hexdigest()[:8]
#     c_func_name = f"comp_graph_openblas_{h}"

    
#     dsl_to_cpp = {}
#     counter = {"matmul": 0, "add": 0, "sub": 0}

#     for mat in all_inputs:
#         dsl_to_cpp[mat] = mat.name

#     for op in ops_sorted:
#         if op in outputs:
#             dsl_to_cpp[op] = op.name
#         else:
#             base = {"matmul": "multiply", "add": "add", "sub": "sub"}.get(op.operations_type)
#             name = f"{base}{counter[op.operations_type]}"
#             counter[op.operations_type] += 1
#             dsl_to_cpp[op] = name

#     code_lines = [
#         '#include <cblas.h>\n',
#         '#include <cstring>\n',
#         '#include <vector>\n',
#         f'extern "C" void {c_func_name}({func_sig}) {{\n',
#         '    std::vector<float*> temps_to_del;\n'
#     ]

    
#     for d in all_dims:
#         code_lines.append(f"    int {d.name}_val = {d.name};\n")

   
#     for op in ops_sorted:
#         if op in all_inputs or op in outputs:
#             continue
#         m = dim_expr(op.shape[0])
#         n = dim_expr(op.shape[1])
#         name = dsl_to_cpp[op]
#         code_lines.append(f"    float* {name} = new float[{m} * {n}]();\n")
#         code_lines.append(f"    temps_to_del.push_back({name});\n")

    
#     for op in ops_sorted:
#         out = dsl_to_cpp[op]
#         a   = dsl_to_cpp[op.inputs[0]]
#         b   = dsl_to_cpp[op.inputs[1]] if len(op.inputs) > 1 else None

#         m = dim_expr(op.shape[0])
#         n = dim_expr(op.shape[1])

#         if op.operations_type == "matmul":
#             k = dim_expr(op.inputs[0].shape[1])
#             code_lines.append(
#                 f"    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, "
#                 f"{m}, {n}, {k}, 1.0f, {a}, {k}, {b}, {n}, 0.0f, {out}, {n});\n"
#             )
#         elif op.operations_type == "add":
#             sz = f"(long long){m} * {n}"
#             code_lines.append(
#                 f"    std::memcpy({out}, {a}, {sz} * sizeof(float));\n"
#                 f"    cblas_saxpy({sz}, 1.0f, {b}, 1, {out}, 1);\n"
#             )
#         elif op.operations_type == "sub":
#             sz = f"(long long){m} * {n}"
#             code_lines.append(
#                 f"    std::memcpy({out}, {a}, {sz} * sizeof(float));\n"
#                 f"    cblas_saxpy({sz}, -1.0f, {b}, 1, {out}, 1);\n"
#             )
#         else:
#             raise NotImplementedError(f"Unsupported op: {op.operations_type}")

    
#     code_lines.append("    for (float* p : temps_to_del) delete[] p;\n")
#     code_lines.append("}\n")

#     return "".join(code_lines), c_func_name, ordered_arg_names






import hashlib
from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.utils import topological_sort_operations, get_graph_io


def dim_expr(d):
    return f"{d.name}_val" if isinstance(d, Var) else str(d)


def compute_openblas(outputs):
    if not outputs:
        return "", "", []

    # Ensure all output nodes have names, or assign temporary ones
    for i, out_node in enumerate(outputs):
        if not getattr(out_node, 'name', None) or out_node.name == "unnamed":
            out_node.name = f"out_{i}"

    ops_sorted = topological_sort_operations(outputs)

    # Collect ALL unique matrix-like nodes (inputs, intermediates, outputs)
    # These will all become float* arguments to the C++ function.
    all_inputs, all_dims = get_graph_io(outputs)
    all_matrix_nodes_in_graph = set(all_inputs)
    for op in ops_sorted:
        all_matrix_nodes_in_graph.add(op)
    for out_node in outputs:
        all_matrix_nodes_in_graph.add(out_node) # Ensure explicit outputs are included

    # --- Assign unique names to intermediate Operation objects ---
    # This loop ensures that every Operation object in the graph has a unique name
    # that will be used in the C++ function signature and for Python-side allocation.
    add_counter = 0
    sub_counter = 0
    matmul_counter = 0
    temp_op_counter = 0 # For any other unsupported intermediate ops

    for op in ops_sorted:
        # If the operation is a final output, its name is already set or will be used as is.
        # Otherwise, assign a unique intermediate name.
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
    # --- End of intermediate naming ---

    # Sort matrix nodes by their (now finalized) name for deterministic argument order in C++ signature
    sorted_matrix_nodes = sorted(list(all_matrix_nodes_in_graph), key=lambda node: node.name)

    func_args = []
    ordered_arg_names = []

    # Add pointers for all matrix-like nodes
    for node in sorted_matrix_nodes:
        func_args.append(f"float* {node.name}")
        ordered_arg_names.append(node.name)

    # Add integer arguments for all symbolic dimensions
    # Sort dimensions by name for deterministic order
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
        '#include <cstring>\n', # Re-added for std::memcpy
        '#include <vector>\n',
        f'extern "C" void {c_func_name}({func_sig}) {{\n',
    ]

    # Declare C++ integer variables for dimensions
    for d in sorted_dims:
        code_lines.append(f"    int {d.name}_val = {d.name};\n")
    code_lines.append("\n")

    # Map DSL node to its C++ variable name (which is now always a function argument)
    # This map is simply a direct mapping from the DSL object's name to itself, as the names are now consistent.
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
            # memcpy is still necessary here because cblas_saxpy is an in-place operation (Y = alpha*X + Y)
            code_lines.append(
                f"    std::memcpy({out}, {a}, {sz} * sizeof(float));\n"
                f"    cblas_saxpy({sz}, 1.0f, {b}, 1, {out}, 1);\n"
            )
        elif op.operations_type == "sub":
            sz = f"(long long){m} * {n}"
            # memcpy is still necessary here because cblas_saxpy is an in-place operation (Y = alpha*X + Y)
            code_lines.append(
                f"    std::memcpy({out}, {a}, {sz} * sizeof(float));\n"
                f"    cblas_saxpy({sz}, -1.0f, {b}, 1, {out}, 1);\n"
            )
        else:
            raise NotImplementedError(f"Unsupported op: {op.operations_type}")

    code_lines.append("}\n")

    return "".join(code_lines), c_func_name, ordered_arg_names




























