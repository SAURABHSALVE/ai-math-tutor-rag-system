# Determinants - Detailed Reference

## 2x2 Determinant
- det([[a,b],[c,d]]) = ad - bc

## 3x3 Determinant
- Expand along any row or column using cofactors
- Sarrus' rule (diagonal method) for quick computation
- det = a(ei-fh) - b(di-fg) + c(dh-eg) for [[a,b,c],[d,e,f],[g,h,i]]

## Properties of Determinants
1. det(A^T) = det(A)
2. Swapping two rows/columns: det changes sign
3. Two identical rows/columns: det = 0
4. Row/column of zeros: det = 0
5. det(kA) = k^n * det(A) for n x n matrix
6. det(AB) = det(A) * det(B)
7. det(A^(-1)) = 1/det(A)
8. Adding multiple of one row to another: det unchanged

## Minors and Cofactors
- Minor M_{ij}: determinant of submatrix after removing row i, column j
- Cofactor C_{ij} = (-1)^(i+j) * M_{ij}
- det(A) = sum of (element * cofactor) along any row or column

## Cramer's Rule
For system Ax = b with det(A) != 0:
- x_i = det(A_i) / det(A)
- A_i = matrix A with column i replaced by b

## Applications
- Area of triangle with vertices (x1,y1), (x2,y2), (x3,y3):
  Area = (1/2)|det([[x1,y1,1],[x2,y2,1],[x3,y3,1]])|
- Three points are collinear iff this determinant = 0
- det(A) = 0 means the system has either no solution or infinitely many
