from matrix import IdentityMatrix, GeneralMatrix, DiagonalMatrix, UpperTriangularMatrix, LowerTriangularMatrix, SymmetricMatrix, ToeplitzMatrix


A=DiagonalMatrix((3,3))
B=UpperTriangularMatrix((3,3))
C=A@B
print(C)