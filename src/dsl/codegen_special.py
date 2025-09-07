from .var import Var
from .codegen_layouts import codegenCpp 
from .utils import get_graph_io, topological_sort_operations
from .operations import Operation 
from .matrix import Matrix

def generate_special_cpp(outs, out_matrix_obj):
    if not outs:
        raise ValueError("No output matrices provided.")

    sortedOps = topological_sort_operations(outs)
    finalExpressionNode = outs[0]

    allGraphNodes = set()
    allInputsFromGraph, allSymDims = get_graph_io(outs)
    
    for node in allInputsFromGraph:
        allGraphNodes.add(node)
    for op in sortedOps:
        allGraphNodes.add(op)
    allGraphNodes.add(out_matrix_obj) 

    allInputsList = list(allInputsFromGraph)
    
    cppCode, orderedCFuncArgNames, cFuncName = codegenCpp(
        finalExpressionNode,
        out_matrix_obj,
        allInputsList
    )
    
    return cppCode, cFuncName, orderedCFuncArgNames
