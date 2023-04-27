from cmb.matrix import load_matrix, load_diagonal_matrix
from cmb.matrix import generate_matrix_D, generate_matrix_Q, matvec_D, matvec_Q
from cmb.utils import *
from procedures.kronecker import kron_aprox
import numpy as np
import time

if __name__ == '__main__':
    # Load matrix A from the file a.csv in data/cmb folder
    A = load_matrix("data/cmb/a.csv")
    # Load diagonal matrix T from the file t.csv in data/cmb folder
    T = load_diagonal_matrix("data/cmb/t.csv")
    # Generate matrix D and Q
    D = generate_matrix_D(3)
    Q = generate_matrix_Q(D, 4)
    # Print the shape of D and Q
    print("D shape: ", D.shape)
    print("Q shape: ", Q.shape)

    

    N = calculate_N_from_level(3)
    n = 4
    x = np.random.rand(N*n)

    # Start a timer
    
    (Q1, Q2) = kron_aprox(Q)
    start = time.time()
    Qx = matvec_Q(Q1, Q2, x)
    # Stop the timer and print the elapsed time
    end = time.time()
    print("Time elapsed: ", end - start)

    start = time.time()
    # Compare Qx to Q.dot(x)
    print("Qx - Q.dot(x): ", np.linalg.norm(Qx - Q.dot(x)))
    end = time.time()
    print("Time elapsed: ", end - start)

    # Assert if Qx is close to Q.dot(x)
    print(Q1)