from enum import Enum 

class LayoutError(Exception):
    pass
class DType(Enum):
    
    float = 1
    int = 2
    double = 3

    
class Layout(Enum):
    AUXILIARY = "auxiliary"
    SCALER = "scaler"
    VECTOR = "vector"
    GENERAL = "general"

    
    DIAGONAL = "diagonal"
    SYMMETRIC = "symmetric"
    LOWER_TRIANGULAR = "lower_triangular"
    UPPER_TRIANGULAR = "upper_triangular"
    TOEPLITZ = "toeplitz"
    IDENTITY = "identity"
    ORTHOGONAL = "orthogonal"
    ZERO = "zero"
    PERMUTATION = "permutation"



layout_properties = list(Layout)
conflicts = {
    prop: set(other for other in layout_properties if other != prop)
    for prop in layout_properties
}



def check_conflicts(props: set[Layout]):
    for prop in props:
        if prop in conflicts:
            conflicting = conflicts[prop].intersection(props)
            if conflicting:
                raise LayoutError(
                    f"{prop.value} conflicts with: {', '.join(p.value for p in conflicting)}"
                )



if __name__ == "__main__":
    pass

