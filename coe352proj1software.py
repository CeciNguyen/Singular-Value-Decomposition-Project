import numpy as np
from numpy import array
import numpy.linalg as na
import scipy.linalg as la 
from scipy.linalg import svd

#Singular-value decomposition
def my_svd(matrix):
    try:
        #Calculate SVD, for test
        #U, S, V = np.linalg.svd(matrix, full_matrices=False)

        #Calculate the eigenvalues and eigenvectors of A.T A
        AtA = matrix.T @ matrix
        eigvals, eigvecs = np.linalg.eig(AtA)

        #Sort the eigenvalues and eigenvectors in descending order
        sortedIndices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sortedIndices]
        eigvecs = eigvecs[:, sortedIndices]

        #Calculate singular values and the matrix U
        singularVals = np.sqrt(eigvals)
        U = matrix @ eigvecs / singularVals

        #Calculate the matrix V
        V = eigvecs

        #Calculate matrix condition number
        condNum = singularVals[0] / singularVals[-1]

        #Calculate the matix inverse using eigenvalue/eigenvector method
        ##eigvals, eigvecs = np.linalg.eig(matrix.T @ matrix) (used for a test)
        if any(np.isclose(singularVals, 0.0)):
            return ValueError("Matrix is singular! It does not have an inverse.")
        
        # Adjust the signs of V to ensure consistency with singular values
        same_sign = np.sign((matrix @ V)[0] * (U @ np.diag(singularVals))[0])
        V = V * same_sign.reshape(1, -1)
        
        #Calculate the matrix inverse using the SVD decomp
        matrix_inverse = V @ np.diag(1.0/singularVals) @ U.T

        return {
            "U": U,
            "S": np.diag(singularVals),
            "V": V.T,
            "Condition Number": condNum,
            "Matrix Inverse": matrix_inverse
        }
    except np.linalg.linalg.LinAlgError:
        return ValueError("Matrix is singular! It does not have an inverse.")
    
# Step 1: User Input
n = int(input("Enter the number of springs/masses: "))
k = np.array([float(input(f"Enter spring constant for spring {i}: ")) for i in range(n)])
m = np.array([float(input(f"Enter mass for mass {i}: ")) for i in range(n)])
boundCond = input("Enter boundary condition (one or two fixed ends): ")

# Step 2: Assemble Stiffness Matrix (K) and Load Vector (F)
def assemble_stiffness_matrix_and_load_vector(n, k, m, boundCond):
    K = np.zeros((n, n))
    F = np.zeros((n, 1))

    for i in range(n):
        K[i, i] = k[i]
        if i > 0:
            K[i, i - 1] = -k[i - 1]
            K[i - 1, i] = -k[i - 1]
        F[i, 0] = m[i] * 9.81  # Assuming gravitational acceleration is 9.81 m/s^2

        # Modify K and F based on boundary conditions
        if boundCond == "one fixed end":
            K[0, :] = 0
            K[:, 0] = 0
            K[0, 0] = 1
            F[0] = 0
        elif boundCond == "two fixed ends":
            K[0, :] = 0
            K[:, 0] = 0
            K[0, 0] = 1
            K[-1, :] = 0
            K[:, -1] = 0
            K[-1, -1] = 1
            F[0] = 0
            F[-1] = 0
    return K, F

# Assemble the stiffness matrix K and load vector F
K, F = assemble_stiffness_matrix_and_load_vector(n, k, m, boundCond)

# Step 3: Solve Ku = F System Using SVD
try:
    result = my_svd(K)

    U = result["U"]
    S = result["S"]
    V = result["V"]

    u = np.linalg.solve(S, U.T @ F)

    # Step 4: Calculate Equilibrium Displacements, Internal Stresses, and Elongations
    equilibrium_displacements = U @ u
    internal_stresses = K @ u
    elongations = np.diag(S) @ V.T @ u

    print("Equilibrium Displacements:")
    print(equilibrium_displacements)
    print("Internal Stresses:")
    print(internal_stresses)
    print("Elongations:")
    print(elongations)

except ValueError as e:
    print(str(e))


