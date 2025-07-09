from enum import Enum
import numpy as np

class Property(Enum):
    DIAGONAL = "diagonal"
    UPPER_TRIANGULAR = "upper_triangular"
    LOWER_TRIANGULAR = "lower_triangular"
    SYMMETRIC = "symmetric"
    TOEPLITZ = "toeplitz"
    GENERAL = "general"

def example_layout(property_type, shape=(4, 4)):
    M, N = shape
    A = np.zeros((M, N))

    if property_type == Property.DIAGONAL:
        for i in range(min(M, N)):
            A[i, i] = 1
    elif property_type == Property.UPPER_TRIANGULAR:
        for i in range(M):
            for j in range(i, N):
                A[i, j] = 1
    elif property_type == Property.LOWER_TRIANGULAR:
        for i in range(M):
            for j in range(0, i+1):
                A[i, j] = 1
    elif property_type == Property.SYMMETRIC:
        for i in range(M):
            for j in range(i, N):
                A[i, j] = A[j, i] = 1
    elif property_type == Property.TOEPLITZ:
        for i in range(M):
            for j in range(N):
                A[i, j] = i - j
    elif property_type == Property.GENERAL:
        A = np.random.rand(M, N)
    
    return A
