

from enum import Enum
import functools

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



add_layout_rules = {
    (Layout.DIAGONAL, Layout.DIAGONAL): Layout.DIAGONAL,
    (Layout.UPPER_TRIANGULAR, Layout.UPPER_TRIANGULAR): Layout.UPPER_TRIANGULAR,
    (Layout.LOWER_TRIANGULAR, Layout.LOWER_TRIANGULAR): Layout.LOWER_TRIANGULAR,
    (Layout.SYMMETRIC, Layout.SYMMETRIC): Layout.SYMMETRIC,
    (Layout.TOEPLITZ, Layout.TOEPLITZ): Layout.TOEPLITZ,
}

for (a, b), result in list(add_layout_rules.items()):
    if (b, a) not in add_layout_rules:
        add_layout_rules[(b, a)] = result

for layout in Layout:
    add_layout_rules[(layout, Layout.GENERAL)] = Layout.GENERAL
    add_layout_rules[(Layout.GENERAL, layout)] = Layout.GENERAL

sub_layout_rules = dict(add_layout_rules)

matmul_layout_rules = {
    (Layout.DIAGONAL, Layout.DIAGONAL): Layout.DIAGONAL,
    (Layout.UPPER_TRIANGULAR, Layout.DIAGONAL): Layout.UPPER_TRIANGULAR,
    (Layout.DIAGONAL, Layout.UPPER_TRIANGULAR): Layout.UPPER_TRIANGULAR,
    (Layout.LOWER_TRIANGULAR, Layout.DIAGONAL): Layout.LOWER_TRIANGULAR,
    (Layout.DIAGONAL, Layout.LOWER_TRIANGULAR): Layout.LOWER_TRIANGULAR,
    (Layout.SYMMETRIC, Layout.DIAGONAL): Layout.SYMMETRIC,
    (Layout.DIAGONAL, Layout.SYMMETRIC): Layout.SYMMETRIC,
    (Layout.TOEPLITZ, Layout.DIAGONAL): Layout.TOEPLITZ,
    (Layout.DIAGONAL, Layout.TOEPLITZ): Layout.TOEPLITZ,

    (Layout.UPPER_TRIANGULAR, Layout.UPPER_TRIANGULAR): Layout.UPPER_TRIANGULAR,
    (Layout.LOWER_TRIANGULAR, Layout.LOWER_TRIANGULAR): Layout.LOWER_TRIANGULAR,
    (Layout.LOWER_TRIANGULAR, Layout.UPPER_TRIANGULAR): Layout.GENERAL,
    (Layout.UPPER_TRIANGULAR, Layout.LOWER_TRIANGULAR): Layout.GENERAL,
    (Layout.TOEPLITZ, Layout.TOEPLITZ): Layout.GENERAL,
}

for layout in Layout:
    matmul_layout_rules[(layout, Layout.GENERAL)] = Layout.GENERAL
    matmul_layout_rules[(Layout.GENERAL, layout)] = Layout.GENERAL

transpose_layout_rules = {
    Layout.DIAGONAL: Layout.DIAGONAL,
    Layout.SYMMETRIC: Layout.SYMMETRIC,
    Layout.UPPER_TRIANGULAR: Layout.LOWER_TRIANGULAR,
    Layout.LOWER_TRIANGULAR: Layout.UPPER_TRIANGULAR,
    Layout.TOEPLITZ: Layout.TOEPLITZ,
    Layout.GENERAL: Layout.GENERAL,
    Layout.IDENTITY: Layout.IDENTITY,
    Layout.ORTHOGONAL: Layout.ORTHOGONAL,
    Layout.ZERO: Layout.ZERO,
    Layout.PERMUTATION: Layout.PERMUTATION,
    Layout.GENERAL: Layout.GENERAL
}
inverse_layout_rules = {
    Layout.DIAGONAL: Layout.DIAGONAL,
    Layout.SYMMETRIC: Layout.SYMMETRIC,
    Layout.UPPER_TRIANGULAR: Layout.UPPER_TRIANGULAR,
    Layout.LOWER_TRIANGULAR: Layout.LOWER_TRIANGULAR,
    Layout.TOEPLITZ: Layout.TOEPLITZ,
    Layout.GENERAL: Layout.GENERAL,
    Layout.IDENTITY: Layout.IDENTITY,
    Layout.ORTHOGONAL: Layout.ORTHOGONAL,
    Layout.ZERO: Layout.ZERO,
    Layout.PERMUTATION: Layout.PERMUTATION
}

def get_layout_result(op: str, left: Layout, right: Layout) -> Layout:
    if op == "add":
        return add_layout_rules.get((left, right), Layout.GENERAL)
    elif op == "sub":
        return sub_layout_rules.get((left, right), Layout.GENERAL)
    elif op == "matmul":
        return matmul_layout_rules.get((left, right), Layout.GENERAL)
    elif op == "transpose":
        return transpose_layout_rules.get(left, Layout.GENERAL)
    elif op == "inverse":
        return inverse_layout_rules.get(left, Layout.GENERAL)
    else:
        raise ValueError(f"Operator '{op}' not supported.")

if __name__ == "__main__":
    pass
