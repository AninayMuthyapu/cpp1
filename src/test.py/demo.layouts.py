from dsl.matrix import *
import numpy as np

def print_matrix(mat):
    print(np.array_str(mat, precision=0, suppress_small=True))

mats = [
    DiagonalMatrix((4, 4)),
    UpperTriangularMatrix((4, 4)),
    LowerTriangularMatrix((4, 4)),
    SymmetricMatrix((4, 4)),
    ToeplitzMatrix((4, 4)),
    GeneralMatrix((4, 4))
]

for m in mats:
    print(f"{m.layout.value.upper()}:\n")
    print_matrix(m.sample())
    print("-" * 30)
