# Files containing functions for matrix-vector multiplication
import numpy as np
import kronecker
import unittest

# Direct multiplication using numpy
def matvec_direct(M, v):
    return np.dot(M, v)

# Function to calculate Ax where A is a matrix that is A = A_1 \otimes \A_2 and x is a vector
# Given A_1 and A_2, this function calculates Ax
def matvec_kron(A1, A2, x):
    x1 = x[:A1.shape[1]]
    x2 = x[A1.shape[1]:]
    y1 = np.dot(A1, x1)
    y2 = np.dot(A2, x2)
    y = np.kron(y1, y2)
    return y

# Write tests to compare running time of matvec_direct and matvec_kron
class TestMatvec(unittest.TestCase):
    def test_matvec(self):
        A = np.random.rand(256, 512)
        A_1, A_2 = kronecker.kron_aprox(A)
        x = np.random.rand(512)
        y1 = matvec_direct(A, x)
        y2 = matvec_kron(A_1, A_2, x)
        self.assertTrue(np.allclose(y1, y2))

# Run the tests
if __name__ == '__main__':
    unittest.main()
