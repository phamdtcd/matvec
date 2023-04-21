# Files containing functions for Kronecker product decomposition
import numpy as np
import unittest

# Permutation operator
def permute(mat, A_dim1, A_dim2, B_dim1, B_dim2):
    ans = np.zeros((A_dim1 * A_dim2, B_dim1 * B_dim2))
    for j in range(A_dim2):
        for i in range(A_dim1):
            ans[A_dim1 * j + i, :] = mat[i * B_dim1 : (i + 1) * B_dim1, 
                                         j * B_dim2 : (j + 1) * B_dim2].reshape(B_dim1 * B_dim2, order = 'F')
    return ans

# Kronecker product decomposition that minimizes the Frobenius norm
def kron_decompose(A, A_dim1, A_dim2, B_dim1, B_dim2):
    X = permute(A, A_dim1, A_dim2, B_dim1, B_dim2)
    u, s, v = np.linalg.svd(X, full_matrices = False)
    C = np.sqrt(s[0]) * u[:, 0].reshape(A_dim1, A_dim2, order = 'F')
    D = np.sqrt(s[0]) * v[0, :].reshape(B_dim1, B_dim2, order = 'F')
    return C, D

# Function that finds the best Kronecker product decomposition
def kron_aprox(A):
    A_dim1, A_dim2 = A.shape
    min_obj = np.inf
    for a in range(1, A_dim1 + 1):
        if A_dim1 % a == 0:
            c = A_dim1 // a
            for b in range(1, A_dim2 + 1):
                if A_dim2 % b == 0:
                    d = A_dim2 // b
                    C, D = kron_decompose(A, a, b, c, d)
                    obj = np.linalg.norm(A - np.kron(C, D), 'fro')
                    if obj < min_obj:
                        min_obj = obj
                        C_best = C
                        D_best = D
    return C_best, D_best

# Write unit tests for kron function
class TestKron(unittest.TestCase):
    def test_kron(self):
        A = np.random.rand(256, 512)
        C, D = kron_aprox(A)
        self.assertTrue(np.allclose(A, np.kron(C, D)))

# Run the tests
if __name__ == '__main__':
    unittest.main()
