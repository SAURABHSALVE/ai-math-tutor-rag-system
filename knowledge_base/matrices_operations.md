# Matrix Operations and Types

## Matrix Types
- Row matrix: 1 x n
- Column matrix: m x 1
- Square matrix: n x n
- Diagonal matrix: a_ij = 0 for i != j
- Scalar matrix: diagonal with all diagonal entries equal
- Identity matrix (I): diagonal with all 1s
- Zero matrix: all entries 0
- Symmetric: A = A^T
- Skew-symmetric: A = -A^T (diagonal must be 0)
- Orthogonal: A^T = A^(-1), i.e., A * A^T = I

## Matrix Operations
- Addition: (A+B)_{ij} = A_{ij} + B_{ij} (same dimensions required)
- Scalar multiplication: (kA)_{ij} = k * A_{ij}
- Matrix multiplication: (AB)_{ij} = sum_k A_{ik} * B_{kj}
  - A(m x n) * B(n x p) = C(m x p)
  - NOT commutative: AB != BA in general
- Transpose: (A^T)_{ij} = A_{ji}
  - (A+B)^T = A^T + B^T
  - (AB)^T = B^T * A^T
  - (A^T)^T = A

## Inverse of a Matrix
- Exists only if det(A) != 0 (non-singular)
- A * A^(-1) = A^(-1) * A = I
- For 2x2: A = [[a,b],[c,d]], A^(-1) = (1/(ad-bc)) * [[d,-b],[-c,a]]
- (AB)^(-1) = B^(-1) * A^(-1)
- (A^T)^(-1) = (A^(-1))^T

## Rank of a Matrix
- Maximum number of linearly independent rows (or columns)
- rank(A) <= min(m, n)
- Full rank: rank = min(m, n)
- rank(AB) <= min(rank(A), rank(B))

## Adjoint (Adjugate)
- adj(A) = transpose of cofactor matrix
- A * adj(A) = det(A) * I
- A^(-1) = adj(A) / det(A)
