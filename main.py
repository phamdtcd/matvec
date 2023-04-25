from cmb.matrix import load_matrix, load_diagonal_matrix
from cmb.matrix import generate_matrix_D, generate_matrix_Q
from kronecker import kron_aprox
if __name__ == '__main__':
    # Load matrix A from the file a.csv in data/cmb folder
    A = load_matrix("data/cmb/a.csv")
    # Load diagonal matrix T from the file t.csv in data/cmb folder
    T = load_diagonal_matrix("data/cmb/t.csv")
    # Generate matrix D and Q
    D = generate_matrix_D(1)
    Q = generate_matrix_Q(D, 4)
    # Print the shape of D and Q
    print("D shape: ", D.shape)
    print("Q shape: ", Q.shape)
    # Test the kron_aprox function on Q
    # Import the kron_aprox function from kronecker.py
    (Q1, Q2) = kron_aprox(Q)

    # Print the shape of Q1 and Q2
    print("Q1 shape: ", Q1.shape)
    print("Q2 shape: ", Q2.shape)