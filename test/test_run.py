from dsl.var import Var
from dsl.matrix import GeneralMatrix
from dsl.runner import run
import numpy as np

A = GeneralMatrix((Var(4), Var(4)))
B = GeneralMatrix((Var(4), Var(4)))
C = A + B

outputs = [C]
inputs = {
    A: np.random.rand(4, 4).astype(np.float32),
    B: np.random.rand(4, 4).astype(np.float32),
}

result = run(outputs, inputs)
print(result)
