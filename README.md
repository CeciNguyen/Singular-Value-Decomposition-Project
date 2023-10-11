# Singular Value Decomposition (SVD) and Structural Analysis

This repository contains python code for performing Singular Value Decomposition (SVD) on a given matrix and using it to solve a structural analysis problem. The code allows for the assembly of stiffness matrices and load vectors for one-dimensional structural systems with different boundary conditions.

## What to Expect?
Within this repositiory, you will find two python scripts. The first script, "SVDfunction.py", contains the personalized SVD function made from scratch. In this python script, there are comments on the bottom with two different test matricies and their results. The second script, "s&msoftwareproj.py", contains the personalized SVD routine and the necessary steps taken to calculate the equilibrium displacments, internal stresses, and elongations of any given spring and mass system.

## 


## Usage of the Software

1. **Installation**: Ensure you have Python and required libraries installed. You can install the necessary libraries using pip:

2. **Run the Software**:
- Open a terminal or command prompt.
- Navigate to the directory containing the software file.
- Run the software using Python:
  ```
  python your_script.py
  ```

3. **Input Parameters**:
- The software will prompt you to enter the following parameters:
  - Number of springs.
  - Boundary conditions, which can be one of the following:
    - 'one_fixed_one_free': One end fixed, one end free.
    - 'two_fixed': Both ends fixed.
    - 'two_free': Both ends free.
  - Spring constants for each spring.
  - Masses between springs, depending on the selected boundary conditions.

4. **Results**:
- The software will calculate the equilibrium displacements, internal stresses, elongations, and the L2 condition number of the stiffness matrix.
- The results will be displayed in the terminal.

## Code Description

### `my_svd` Function
- `my_svd` performs Singular Value Decomposition (SVD) on an input matrix.
- It calculates the left singular vectors, singular values, right singular vectors, condition number, and matrix inverse using the SVD decomposition.
- It raises a `ValueError` if the input matrix is singular.

### `calculate_system` Function
- `calculate_system` analyzes a mechanical system based on the provided input parameters.
- It constructs the stiffness matrix (K) and the force vector (F) for the system.
- It uses the `my_svd` function to calculate the L2 condition number of K and raises an error if K is singular.
- It computes the internal stresses, equilibrium displacements, and elongations for the system.
- It provides robust error handling and helpful error messages.

## Example Usage

```python
num_springs = 4
boundary_conditions = 'two_fixed'
spring_constants = [1, 2, 3, 4]
masses = [1, 2, 3]  # The number of mass values depends on the boundary conditions.

u, F, elongations, l2_condition_number = calculate_system(num_springs, spring_constants, masses, boundary_conditions)

print("Equilibrium Displacements:", u)
print("Internal Stresses:", F)
print("Elongations:", elongations)
print(f"L2 Condition Number of K: {l2_condition_number}")

