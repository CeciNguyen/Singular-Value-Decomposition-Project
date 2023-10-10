import numpy as np
from numpy import array
import numpy.linalg as na
import scipy.linalg as la 
from scipy.linalg import svd

#Singular-value decomposition
def my_svd(matrix, threshold=1e-10) -> dict:
    """
    Perform Singular Value Decomposition (SVD) on the input matrix.
    Args:
        matrix (numpy.ndarray): The input matrix to be decomposed.
        threshold (float, optional): A threshold value to determine if singular values are close to zero.
            Singular values smaller than this threshold will be considered as zero.
            Default is 1e-10.
    Returns:
        dict: A dictionary containing the following components:
            - "U" (numpy.ndarray): The left singular vectors of the input matrix.
            - "S" (numpy.ndarray): A diagonal matrix containing the singular values.
            - "V" (numpy.ndarray): The right singular vectors of the input matrix (transpose of VT).
            - "Condition Number" (float): The condition number of the matrix based on singular values.
            - "Matrix Inverse" (numpy.ndarray): The matrix inverse computed using the SVD decomposition.
    Raises:
        ValueError: If the input matrix is singular (contains zero or near-zero singular values).
        np.linalg.linalg.LinAlgError: If a linear algebra error occurs during computation.
    """
    try:
        #Calculate the eigenvalues and eigenvectors of A.T A
        AtA = matrix.T @ matrix
        eigvals, eigvecs = np.linalg.eig(AtA)

        # Check for singular matrix
        if any(np.isclose(eigvals, 0.0, atol=threshold)):
            return ValueError("Matrix is singular! It does not have an inverse.")

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

        # Apply threshold to singular values
        singularVals[singularVals < threshold] = 0.0

        # Adjust the signs of V to ensure consistency with singular values
        same_sign = np.sign((matrix @ V[:, :len(singularVals)]) @ singularVals)
        V[:, :len(singularVals)] = V[:, :len(singularVals)] * same_sign

        #Calculate the matrix inverse using the SVD decomp
        matrix_inverse = V[:, :len(singularVals)] @ np.diag(1.0 / singularVals) @ U.T

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
def assemble_stiffness_matrix_and_load_vector(n, k, m, boundCond) -> tuple:
    """
    Assemble the stiffness matrix and load vector for a one-dimensional structural problem.
    Args:
        n (int): The number of nodes or elements in the structural system.
        k (numpy.ndarray): An array of stiffness values for each element or node.
        m (numpy.ndarray): An array of mass values for each element or node.
        boundCond (str): Specifies the boundary conditions of the problem.
            - "one fixed end": One end of the system is fixed (zero displacement).
            - "two fixed ends": Both ends of the system are fixed.
    Returns:
        tuple: A tuple containing the stiffness matrix (K) and the load vector (F).
    """
    K = np.zeros((n, n)) #these lines initalize the stiffness matrix 'K' and load vector 'F'. The dimensions of these arrays are determined by the number of nodes 'n'
    F = np.zeros((n, 1))

    for i in range(n): #This loop iterates over all nodes or elements in the finite element model, where n is the total number of nodes.
        K[i, i] = k[i] #This line assigns the stiffness value k[i] to the diagonal element of the stiffness matrix K at position (i, i). Each element of the diagonal corresponds to the stiffness of the individual elements or nodes.
        if i > 0: #This condition checks whether the current node is not the first node (i.e., it's not the leftmost end of the structure). It ensures that the following lines are only executed for nodes other than the first one.
            K[i, i - 1] = -k[i - 1] #These lines set the off-diagonal elements of the stiffness matrix K. They represent the negative stiffness values connecting adjacent nodes. 
            K[i - 1, i] = -k[i - 1]
        F[i, 0] = m[i] * 9.81  # Assuming gravitational acceleration is 9.81 m/s^2. This line calculates the force component in the load vector F at position (i, 0) based on the mass m[i] and gravitational acceleration (9.81 m/s²)

        # Modify K and F based on boundary conditions
        if boundCond == "one fixed end":
            K[0, :] = 0 #These lines set all the elements in the first row and first column of the stiffness matrix K to zero. This effectively fixes the displacement (prevents movement) at the first node or end of the structure.
            K[:, 0] = 0
            K[0, 0] = 1 #This line sets the top-left element of the stiffness matrix K to 1. This is often used to represent a fixed support, where there is no translation or rotation at the first node.
            F[0] = 0    #This line sets the force at the first node (usually associated with the fixed end) to zero. Since the structure is fixed at this end, there is no external force applied to it.
        elif boundCond == "two fixed ends":
            K[0, :] = 0 #These lines, as in the previous case, set all the elements in the first row and first column of the stiffness matrix K to zero.
            K[:, 0] = 0
            K[0, 0] = 1 #This line set the top-left and bottom-right elements of the stiffness matrix K to 1. This represents fixed supports at both ends of the structure.
            K[-1, :] = 0 #These lines set all the elements in the last row and last column of the stiffness matrix K to zero.
            K[:, -1] = 0
            K[-1, -1] = 1 #This line set the top-left and bottom-right elements of the stiffness matrix K to 1. This represents fixed supports at both ends of the structure.
            F[0] = 0    #These lines set the forces at both the first and last nodes to zero. Since the structure is fixed at both ends, there are no external forces applied to it at either end.
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


