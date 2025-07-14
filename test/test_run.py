from dsl.var import Var
from dsl.runner import run
from dsl.matrix import GeneralMatrix
import numpy as np

M = Var("M")
N = Var("N")

A = GeneralMatrix((M, N))
B = GeneralMatrix((M, N))
C = A + B
C.name = "C"

inputs = {
    "A": np.random.rand(4, 4),
    "B": np.random.rand(4, 4),
    "M": 4,
    "N": 4,
}

result = run([C], inputs)
print(result["C"])



