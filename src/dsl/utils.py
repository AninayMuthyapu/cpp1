import sys
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
           
            for input_node in node.operands:
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
            
            for input_node in node.operands:
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
        if hasattr(node, 'shape'):
            for dim in node.shape:
                if isinstance(dim, Var) or isinstance(dim, (int, float)):
                    symbolic_dims.add(dim)
        if isinstance(node, Matrix) and not isinstance(node, Operation):
            input_matrices.add(node)
        elif isinstance(node, Operation):
            
            for input_node in node.operands:
                stack.append(input_node)
    sorted_input_matrices = sorted(list(input_matrices), key=lambda m: m.name or "")
    
    
    filtered_symbolic_vars = []
    for dim in symbolic_dims:
        if isinstance(dim, Var):
            filtered_symbolic_vars.append(dim)
            
    sorted_symbolic_dims = sorted(filtered_symbolic_vars, key=lambda v: str(v.name))
    
    return sorted_input_matrices, sorted_symbolic_dims