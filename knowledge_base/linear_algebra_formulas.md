# Linear Algebra Basics

## Matrices
- Addition: (A+B)_{ij} = A_{ij} + B_{ij} (same dimensions)
- Scalar multiplication: (cA)_{ij} = c * A_{ij}
- Matrix multiplication: (AB)_{ij} = sum_k A_{ik} * B_{kj}
- AB != BA in general (not commutative)
- (AB)^T = B^T * A^T
- (AB)^(-1) = B^(-1) * A^(-1)

## Determinants
### 2x2 Matrix
- det([[a,b],[c,d]]) = ad - bc

### 3x3 Matrix
- Expand along any row or column using cofactors
- det(A) = a11*C11 + a12*C12 + a13*C13

### Properties
- det(AB) = det(A) * det(B)
- det(A^T) = det(A)
- det(kA) = k^n * det(A) for n x n matrix
- If any row/column is all zeros, det = 0
- Swapping two rows changes sign of det
- If two rows are identical, det = 0

## Inverse of a Matrix
- A * A^(-1) = I
- For 2x2: A^(-1) = (1/det(A)) * [[d,-b],[-c,a]]
- A is invertible iff det(A) != 0
- Using row reduction: [A | I] -> [I | A^(-1)]

## Systems of Linear Equations
- Ax = b
- Cramer's rule: x_i = det(A_i) / det(A)
- Consistent if rank(A) = rank([A|b])
- Unique solution if rank(A) = number of unknowns
- Infinite solutions if rank(A) < number of unknowns

## Eigenvalues and Eigenvectors
- Av = lambda * v (v != 0)
- Characteristic equation: det(A - lambda*I) = 0
- Sum of eigenvalues = trace(A)
- Product of eigenvalues = det(A)

## Vectors
- Dot product: a . b = |a||b|cos(theta) = a1*b1 + a2*b2 + a3*b3
- Cross product: a x b = |a||b|sin(theta) * n_hat
- |a x b| = area of parallelogram
- a . (b x c) = scalar triple product = volume of parallelepiped
- Projection of a on b: (a . b / |b|^2) * b
