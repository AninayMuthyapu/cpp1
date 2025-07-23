
import hashlib
from dsl.utils import topological_sort_operations, get_graph_io
from dsl.var import Var


def dim_expr(d):
    return f"{d.name}_val" if isinstance(d, Var) else str(d)


def compute_openblas(outputs):
    if not outputs:
        return "", "", []

    
    for i, out in enumerate(outputs):
        if not getattr(out, 'name', None) or out.name == "unnamed":
            out.name = f"out_{i}"

    ops_sorted = topological_sort_operations(outputs)
    all_inputs, all_dims = get_graph_io(outputs)

    
    func_args, ordered_arg_names = [], []
    for m in all_inputs:
        func_args.append(f"float* {m.name}")
        ordered_arg_names.append(m.name)
    for out in outputs:
        func_args.append(f"float* {out.name}")
    for d in all_dims:
        func_args.append(f"int {d.name}")
        ordered_arg_names.append(d.name)
    func_sig = ", ".join(func_args)

    sig_str = str(ordered_arg_names) + str([op.operations_type for op in ops_sorted])
    h = hashlib.md5(sig_str.encode()).hexdigest()[:8]
    c_func_name = f"comp_graph_openblas_{h}"

    
    dsl_to_cpp = {}
    counter = {"matmul": 0, "add": 0, "sub": 0}
    for m in all_inputs:
        dsl_to_cpp[m] = m.name
    for op in ops_sorted:
        base = {"matmul": "multiply", "add": "add", "sub": "sub"}.get(op.operations_type)
        name = f"{base}{counter[op.operations_type]}"
        counter[op.operations_type] += 1
        dsl_to_cpp[op] = name

    
    code_lines = [
        '#include <cblas.h>\n',
        '#include <cstring>\n',
        '#include <vector>\n',
        f'extern "C" void {c_func_name}({func_sig}) {{\n',
        '    std::vector<float*> temps_to_del;\n'
    ]
    for d in all_dims:
        code_lines.append(f"    int {d.name}_val = {d.name};\n")

    
    for op in ops_sorted:
        m = dim_expr(op.shape[0])
        n = dim_expr(op.shape[1])
        name = dsl_to_cpp[op]
        code_lines.append(f"    float* {name} = new float[{m} * {n}]();\n")
        code_lines.append(f"    temps_to_del.push_back({name});\n")

    
    for op in ops_sorted:
        out = dsl_to_cpp[op]
        a = dsl_to_cpp[op.inputs[0]]
        b = dsl_to_cpp[op.inputs[1]] if len(op.inputs) > 1 else None
        m = dim_expr(op.shape[0])
        n = dim_expr(op.shape[1])

        if op.operations_type == "matmul":
            k = dim_expr(op.inputs[0].shape[1])  
            code_lines.append(
                f"    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, "
                f"{m}, {n}, {k}, 1.0f, "
                f"{a}, {k}, "
                f"{b}, {n}, "
                f"0.0f, {out}, {n});\n"
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

    
    for op in outputs:
        m = dim_expr(op.shape[0])
        n = dim_expr(op.shape[1])
        code_lines.append(
            f"    std::memcpy({op.name}, {dsl_to_cpp[op]}, {m} * {n} * sizeof(float));\n"
        )

    code_lines.append("    for (float* p : temps_to_del) delete[] p;\n")
    code_lines.append("}\n")

    return "".join(code_lines), c_func_name, ordered_arg_names

























# import hashlib
# from dsl.utils import topological_sort_operations, get_graph_io
# from dsl.var import Var

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
#         a = dsl_to_cpp[op.inputs[0]]
#         b = dsl_to_cpp[op.inputs[1]] if len(op.inputs) > 1 else None

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


