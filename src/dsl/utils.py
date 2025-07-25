from .matrix import Matrix
from .operations import Operation
from .var import Var

def var_names(self, frame_locals):
    for name, var in frame_locals.items():
        if isinstance(var, Matrix):
            if not hasattr(var, 'name') or var.name == "unnamed" or var.name is None:
                var.name = name

def topological_sort_operations(output_nodes):
    visited = set()
    recursion_stack = set()
    sorted_list = []

    def dfs_visit(node):
        if node in recursion_stack:
            raise ValueError("Cycle detected in the DSL graph. ")
        if node in visited:
            return
        recursion_stack.add(node)
        if isinstance(node, Operation):
            for input_node in node.inputs:
                dfs_visit(input_node)
        recursion_stack.remove(node)
        visited.add(node)
        if isinstance(node, Operation):
            sorted_list.append(node)

    all_operations_in_graph = set()
    temp_stack = list(output_nodes)
    temp_visited = set()
    while temp_stack:
        node = temp_stack.pop()
        if node in temp_visited:
            continue
        temp_visited.add(node)
        if isinstance(node, Operation):
            all_operations_in_graph.add(node)
            for input_node in node.inputs:
                temp_stack.append(input_node)
    sorted_unique_operations = sorted(list(all_operations_in_graph), 
                                      key=lambda op: op.name if hasattr(op, 'name') and op.name else str(id(op)))
    for op_node in sorted_unique_operations:
        if op_node not in visited:
            dfs_visit(op_node)
    return sorted_list

def get_graph_io(output_nodes):
    input_matrices = set()
    symbolic_dims = set()
    visited_nodes = set()
    stack = list(output_nodes)
    while stack:
        node = stack.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        for dim in node.shape:
            if isinstance(dim, Var):
                symbolic_dims.add(dim)
        if isinstance(node, Matrix) and not isinstance(node, Operation):
            input_matrices.add(node)
        elif isinstance(node, Operation):
            for input_node in node.inputs:
                stack.append(input_node)
    sorted_input_matrices = sorted(list(input_matrices), key=lambda m: m.name or "")
    sorted_symbolic_dims = sorted(list(symbolic_dims), key=lambda v: v.name)
    return sorted_input_matrices, sorted_symbolic_dims
