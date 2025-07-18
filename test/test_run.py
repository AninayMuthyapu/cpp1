from dsl.var import Var
from dsl.runner import run  # unified runner.py
from dsl.matrix import GeneralMatrix
import numpy as np

# Define symbolic dimensions
M = Var("M")
N = Var("N")
K = Var("K")

# Define matrices
A = GeneralMatrix((M, K))
B = GeneralMatrix((K, N))
C = A + B
C.name = "C"

# Input data
 # For reproducibility
inputs = {
    "A": np.random.rand(4, 4).astype(np.float32),
    "B": np.random.rand(4, 4).astype(np.float32),
    "M": 4,
    "N": 4,
    "K": 4
}

# Choose backend explicitly: "openblas" or "numpy"
result = run([C], inputs, backend="openblas")
result2 = run([C], inputs, backend="cpp") 
result3 = run([C], inputs, backend="numpy")


# Print result
print("Computed C (OpenBLAS):\n", result["C"])
print("Computed C cpp:\n", result2["C"])
print("Computed C numpy:\n", result3["C"])
