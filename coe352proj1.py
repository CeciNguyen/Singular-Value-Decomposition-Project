import numpy as np
from numpy import array
import numpy.linalg as na
import scipy.linalg as la 
from scipy.linalg import svd

#Singular-value decomposition
def my_svd(matrix, threshold=1e-10):
    """
    Doc string
    """
    try:
        #Calculate SVD, for test
        #U, S, V = np.linalg.svd(matrix, full_matrices=False)

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

        #Calculate the matix inverse using eigenvalue/eigenvector method
        ##eigvals, eigvecs = np.linalg.eig(matrix.T @ matrix) (used for a test)
        #if any(np.isclose(singularVals, 0.0)):
            #return ValueError("Matrix is singular! It does not have an inverse.")
        
        # Adjust the signs of V to ensure consistency with singular values
        #same_sign = np.sign((matrix @ V)[0] * (U @ np.diag(singularVals))[0])
        #V = V * same_sign.reshape(1, -1)
        same_sign = np.sign((matrix @ V[:, :len(singularVals)]) @ singularVals)
        V[:, :len(singularVals)] = V[:, :len(singularVals)] * same_sign

        #Calculate the matrix inverse using the SVD decomp
        #matrix_inverse = V @ np.diag(1.0/singularVals) @ U.T
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
    
#Call the function!
if __name__ == "__main__":
    #Create a sample matrix 
    SampMatrix = np.array([[3,2,3],
                           [2,3,-2],
                           [1,2,2]])
    
        #Print the result
    result = my_svd(SampMatrix)
    #for key, value in result.items():
    #    print(f"{key}:\n{value}\n")

    # Print the result
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"{key}:\n{value}\n")
    else:
        print(result)
        
    #Python svd built-in function
    U, S, VT = svd(SampMatrix)
    print("Matrix U: \n")
    print(U)
    print("\nMatrix S:\n")
    print(S)
    print("\nMatrix VT:\n")
    print(VT)

###Results of testing SampMatrix = np.array([[3,2,3],[2,3,-2],])
#Mysvd function
#Matrix is singular! It does not have an inverse.

#Builtin svd function
#Matrix U: 
#[[-0.83205029 -0.5547002 ]
# [-0.5547002   0.83205029]]
#Matrix S:
#[5.09901951 3.60555128]
#Matrix VT:
#[[-7.07106781e-01 -6.52713952e-01 -2.71964147e-01]
# [ 9.89578804e-17  3.84615385e-01 -9.23076923e-01]
# [-7.07106781e-01  6.52713952e-01  2.71964147e-01]]

