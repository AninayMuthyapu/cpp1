
from enum import Enum, auto


class Layout:
  
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Layout({self.name})"


general_layout = Layout("general")
diagonal_layout = Layout("diagonal")
upper_triangular_layout = Layout("upper_triangular")
lower_triangular_layout = Layout("lower_triangular")
symmetric_layout = Layout("symmetric")


def check_conflicts(layout1, layout2):
   
    return True


class DType(Enum):
    
    float = auto()
    double = auto()
    int = auto()
    

















# from .layout import Layout, LayoutError


# add_layout_rules = {
#     (Layout.DIAGONAL, Layout.DIAGONAL): Layout.DIAGONAL,
#     (Layout.UPPER_TRIANGULAR, Layout.UPPER_TRIANGULAR): Layout.UPPER_TRIANGULAR,
#     (Layout.LOWER_TRIANGULAR, Layout.LOWER_TRIANGULAR): Layout.LOWER_TRIANGULAR,
#     (Layout.SYMMETRIC, Layout.SYMMETRIC): Layout.SYMMETRIC,
#     (Layout.TOEPLITZ, Layout.TOEPLITZ): Layout.TOEPLITZ,
# }


# for (a, b), result in list(add_layout_rules.items()):
#     if (b, a) not in add_layout_rules:
#         add_layout_rules[(b, a)] = result


# for layout in Layout:
#     add_layout_rules[(layout, Layout.GENERAL)] = Layout.GENERAL
#     add_layout_rules[(Layout.GENERAL, layout)] = Layout.GENERAL


# sub_layout_rules = dict(add_layout_rules)



# matmul_layout_rules = {
#     (Layout.DIAGONAL, Layout.DIAGONAL): Layout.DIAGONAL,
#     (Layout.UPPER_TRIANGULAR, Layout.DIAGONAL): Layout.UPPER_TRIANGULAR,
#     (Layout.DIAGONAL, Layout.UPPER_TRIANGULAR): Layout.UPPER_TRIANGULAR,
#     (Layout.LOWER_TRIANGULAR, Layout.DIAGONAL): Layout.LOWER_TRIANGULAR,
#     (Layout.DIAGONAL, Layout.LOWER_TRIANGULAR): Layout.LOWER_TRIANGULAR,
#     (Layout.SYMMETRIC, Layout.DIAGONAL): Layout.SYMMETRIC,
#     (Layout.DIAGONAL, Layout.SYMMETRIC): Layout.SYMMETRIC,
#     (Layout.TOEPLITZ, Layout.DIAGONAL): Layout.TOEPLITZ,
#     (Layout.DIAGONAL, Layout.TOEPLITZ): Layout.TOEPLITZ,

#     (Layout.UPPER_TRIANGULAR, Layout.UPPER_TRIANGULAR): Layout.UPPER_TRIANGULAR,
#     (Layout.LOWER_TRIANGULAR, Layout.LOWER_TRIANGULAR): Layout.LOWER_TRIANGULAR,
#     (Layout.LOWER_TRIANGULAR, Layout.UPPER_TRIANGULAR): Layout.GENERAL,
#     (Layout.UPPER_TRIANGULAR, Layout.LOWER_TRIANGULAR): Layout.GENERAL,
#     (Layout.TOEPLITZ, Layout.TOEPLITZ): Layout.GENERAL,
# }


# for layout in Layout:
#     matmul_layout_rules[(layout, Layout.GENERAL)] = Layout.GENERAL
#     matmul_layout_rules[(Layout.GENERAL, layout)] = Layout.GENERAL


# transpose_layout_rules = {
#     Layout.DIAGONAL: Layout.DIAGONAL,
#     Layout.SYMMETRIC: Layout.SYMMETRIC,
#     Layout.UPPER_TRIANGULAR: Layout.LOWER_TRIANGULAR,
#     Layout.LOWER_TRIANGULAR: Layout.UPPER_TRIANGULAR,
#     Layout.TOEPLITZ: Layout.TOEPLITZ,
#     Layout.GENERAL: Layout.GENERAL,
#     Layout.IDENTITY: Layout.IDENTITY,
#     Layout.ORTHOGONAL: Layout.ORTHOGONAL,
#     Layout.ZERO: Layout.ZERO,
#     Layout.PERMUTATION: Layout.PERMUTATION,
#     Layout.GENERAL: Layout.GENERAL
# }
# inverse_layout_rules = {
#     Layout.DIAGONAL: Layout.DIAGONAL,
#     Layout.SYMMETRIC: Layout.SYMMETRIC,
#     Layout.UPPER_TRIANGULAR: Layout.UPPER_TRIANGULAR,
#     Layout.LOWER_TRIANGULAR: Layout.LOWER_TRIANGULAR,
#     Layout.TOEPLITZ: Layout.TOEPLITZ,
#     Layout.GENERAL: Layout.GENERAL,
#     Layout.IDENTITY: Layout.IDENTITY,
#     Layout.ORTHOGONAL: Layout.ORTHOGONAL,
#     Layout.ZERO: Layout.ZERO,
#     Layout.PERMUTATION: Layout.PERMUTATION
# }

    



# def get_layout_result(op: str, left: Layout, right: Layout) -> Layout:
#     if op == "+":
#         return add_layout_rules.get((left, right), Layout.GENERAL)
#     elif op == "-":
#         return sub_layout_rules.get((left, right), Layout.GENERAL)
#     elif op == "@":
#         return matmul_layout_rules.get((left, right), Layout.GENERAL)
#     elif op == "transpose":
#         return transpose_layout_rules.get(left, Layout.GENERAL)
#     elif op == "inverse":
#         return inverse_layout_rules.get(left, Layout.GENERAL)
#     else:
#         raise ValueError(f"Operator '{op}' not supported.")




# layout_rules.py