from .layout import Layout, LayoutError


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



def get_layout_result(op: str, left: Layout, right: Layout) -> Layout:
    if op == "+":
        return add_layout_rules.get((left, right), Layout.GENERAL)
    elif op == "-":
        return sub_layout_rules.get((left, right), Layout.GENERAL)
    elif op == "@":
        return matmul_layout_rules.get((left, right), Layout.GENERAL)
    else:
        raise ValueError(f"Operator '{op}' not supported.")
