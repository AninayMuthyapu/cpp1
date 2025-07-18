from dsl.var import Var
from dsl.runner import run  
from dsl.matrix import GeneralMatrix
import numpy as np


M = Var("M")
N = Var("N")
K = Var("K")

A = GeneralMatrix((M, K))
B = GeneralMatrix((K, N))
C = A + B
C.name = "C"


inputs = {
    "A": np.random.rand(4, 4).astype(np.float32),
    "B": np.random.rand(4, 4).astype(np.float32),
    "M": 4,
    "N": 4,
    "K": 4
}


result = run([C], inputs, backend="openblas")
result2 = run([C], inputs, backend="cpp") 
result3 = run([C], inputs, backend="numpy")



print("Computed C (OpenBLAS):\n", result["C"])
print("Computed C cpp:\n", result2["C"])
print("Computed C numpy:\n", result3["C"])
