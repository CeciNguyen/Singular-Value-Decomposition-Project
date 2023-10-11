import numpy as np
from numpy import array
import numpy.linalg as na
import scipy.linalg as la 
from scipy.linalg import svd

#Singular-value decomposition
def my_svd(matrix, threshold=1e-15) -> dict:
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
            raise ValueError("Matrix is singular! It does not have an inverse.")

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
        raise ValueError("Matrix is singular! It does not have an inverse.")
    
def calculate_system(num_springs, spring_constants, masses, boundary_conditions):
    """
    Calculate the equilibrium displacements, internal stresses, elongations, and L2 condition number
    of a mechanical system composed of springs, masses, and specified boundary conditions.
    Args:
        num_springs (int): The number of springs in the system.
        spring_constants (list of float): A list of spring constants for each spring.
        masses (list of float): A list of masses for the masses between springs.
        boundary_conditions (str): The boundary conditions of the system, which can be one of:
            - 'one_fixed_one_free': One end fixed, one end free.
            - 'two_fixed': Both ends fixed.
            - 'two_free': Both ends free.
    Returns:
        tuple: A tuple containing the following components:
            - u (numpy.ndarray): An array of equilibrium displacements for each node.
            - F (numpy.ndarray): An array of internal stresses at each node.
            - elongations (numpy.ndarray): An array of elongations for each spring.
            - l2_condition_number (float): The L2 condition number of the stiffness matrix.
    """
    try:
        # Initialize system parameters
        num_nodes = num_springs + 1
        K = np.zeros((num_nodes, num_nodes))
        F = np.zeros(num_nodes)
        u = np.zeros(num_nodes)
        
        # Define boundary conditions
        if boundary_conditions == 'one_fixed_one_free':
            u[0] = 0  # Fixed end
        elif boundary_conditions == 'two_fixed':
            u[0] = 0  # Fixed end
            u[-1] = 0  # Fixed end
        elif boundary_conditions == 'two_free':
            pass  # Both ends are free
        
        # Construct the stiffness matrix K and the force vector F
        for i in range(num_springs-1):
            K[i, i] += spring_constants[i] + spring_constants[i + 1]
            K[i, i + 1] = -spring_constants[i + 1]
            K[i + 1, i] = -spring_constants[i + 1]
            
            
            # Calculate the condition number of K using SVD
            svd_result = my_svd(K)
            if svd_result is None:
                raise ValueError("Matrix is singular! It does not have an inverse.")
            
            # Extract the condition number from the svd_result
            l2_condition_number = svd_result["Condition Number"]

            # Calculate internal stresses and equilibrium displacements
            F = masses * 9.81
            u = svd_result["Matrix Inverse"] @ F
            
            # Calculate elongations
            elongations = np.zeros(num_springs)
            for i in range(num_springs):
                elongations[i] = (u[i + 1] - u[i]) * spring_constants[i]
    
    except Exception as e:
        # Handle any exceptions and provide helpful error messages
        print("Error:", str(e))
        return None, None, None, None

    return u, F, elongations, l2_condition_number

# User input
num_springs = int(input("Enter the number of springs: "))
boundary_conditions = input("Enter boundary conditions (one_fixed_one_free, two_fixed, or two_free): ")
spring_constants = [float(input(f"Enter spring constant for spring {i}: ")) for i in range(num_springs)]
masses = []
if boundary_conditions == "one_fixed_one_free":
    masses = [float(input(f"Enter mass for mass {i}: ")) for i in range(num_springs)]
elif boundary_conditions == "two_fixed":
    masses = [float(input(f"Enter mass for mass {i}: ")) for i in range(num_springs-1)]
elif boundary_conditions == "two_free":
    masses = [float(input(f"Enter mass for mass {i}: ")) for i in range(num_springs+1)]


u, F, elongations, l2_condition_number = calculate_system(num_springs, spring_constants, masses, boundary_conditions)

#Print results
print("Equilibrium Displacements:", u)
print("Internal Stresses:", F)
print("Elongations:", elongations)
print(f"L2 Condition Number of K: {l2_condition_number}")