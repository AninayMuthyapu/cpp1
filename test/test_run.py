from dsl.var import Var
from dsl.runner import run
from dsl.matrix import GeneralMatrix
import numpy as np
from dsl.utils import var_names

M = Var("M")
N = Var("N")
P = Var("P")
K = Var("K")

A = GeneralMatrix((M, P))
B = GeneralMatrix((M, N))
C = GeneralMatrix((N, P))

var_names(None, locals())

Out = A + (B @ C)


inputs = {
    "A": np.random.randint(0, 100, (M, P)).astype(np.float32), 
    "B": np.random.randint(0, 100, (M, N)).astype(np.float32),
    "C": np.random.randint(0, 100, (N, P)).astype(np.float32), 
    "M": 32,
    "N": 16,
    "P": 24,
    "K": 32,
}


result_numpy = run([Out], inputs, backend="numpy")
Out_numpy = result_numpy[Out.name]

print("\nComputed Out (NumPy):\n", Out_numpy)


result_openblas = run([Out], inputs, backend="openblas")
Out_openblas = result_openblas[Out.name]



result_cpp = run([Out], inputs, backend="cpp")
Out_cpp = result_cpp[Out.name]
print("Computed  (C++):\n", Out_cpp)

