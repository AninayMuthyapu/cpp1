from .properties import Property

# --- Addition and Subtraction Rules ---
add_layout_rules = {
    (Property.DIAGONAL, Property.DIAGONAL): Property.DIAGONAL,
    (Property.UPPER_TRIANGULAR, Property.UPPER_TRIANGULAR): Property.UPPER_TRIANGULAR,
    (Property.LOWER_TRIANGULAR, Property.LOWER_TRIANGULAR): Property.LOWER_TRIANGULAR,
    (Property.SYMMETRIC, Property.SYMMETRIC): Property.SYMMETRIC,
    (Property.TOEPLITZ, Property.TOEPLITZ): Property.TOEPLITZ,
}

# commutativity (A + B == B + A)
for (a, b), result in list(add_layout_rules.items()):
    if (b, a) not in add_layout_rules:
        add_layout_rules[(b, a)] = result

# General 
for layout in Property:
    add_layout_rules[(layout, Property.GENERAL)] = Property.GENERAL
    add_layout_rules[(Property.GENERAL, layout)] = Property.GENERAL

# Subtraction uses same rules as addition
sub_layout_rules = dict(add_layout_rules)


# --- Matrix Multiplication Rules ---
matmul_layout_rules = {
    (Property.DIAGONAL, Property.DIAGONAL): Property.DIAGONAL,
    (Property.UPPER_TRIANGULAR, Property.DIAGONAL): Property.UPPER_TRIANGULAR,
    (Property.DIAGONAL, Property.UPPER_TRIANGULAR): Property.UPPER_TRIANGULAR,
    (Property.LOWER_TRIANGULAR, Property.DIAGONAL): Property.LOWER_TRIANGULAR,
    (Property.DIAGONAL, Property.LOWER_TRIANGULAR): Property.LOWER_TRIANGULAR,
    (Property.SYMMETRIC, Property.DIAGONAL): Property.SYMMETRIC,
    (Property.DIAGONAL, Property.SYMMETRIC): Property.SYMMETRIC,
    (Property.TOEPLITZ, Property.DIAGONAL): Property.TOEPLITZ,
    (Property.DIAGONAL, Property.TOEPLITZ): Property.TOEPLITZ,

    (Property.UPPER_TRIANGULAR, Property.UPPER_TRIANGULAR): Property.UPPER_TRIANGULAR,
    (Property.LOWER_TRIANGULAR, Property.LOWER_TRIANGULAR): Property.LOWER_TRIANGULAR,
    (Property.LOWER_TRIANGULAR, Property.UPPER_TRIANGULAR): Property.GENERAL,
    (Property.UPPER_TRIANGULAR, Property.LOWER_TRIANGULAR): Property.GENERAL,
    (Property.TOEPLITZ, Property.TOEPLITZ): Property.GENERAL,
}


for layout in Property:
    matmul_layout_rules[(layout, Property.GENERAL)] = Property.GENERAL
    matmul_layout_rules[(Property.GENERAL, layout)] = Property.GENERAL



def get_layout_result(op: str, left: Property, right: Property) -> Property:
    if op == "+":
        return add_layout_rules.get((left, right), Property.GENERAL)
    elif op == "-":
        return sub_layout_rules.get((left, right), Property.GENERAL)
    elif op == "@":
        return matmul_layout_rules.get((left, right), Property.GENERAL)
    else:
        raise ValueError(f"Operator '{op}' not supported.")
