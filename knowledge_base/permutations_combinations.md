# Permutations and Combinations

## Fundamental Counting Principle
- If task A can be done in m ways and task B in n ways:
  - Both (A then B): m * n ways
  - Either (A or B): m + n ways (if mutually exclusive)

## Permutations (Order Matters)
- nPr = n! / (n-r)!
- Arranging n distinct objects: n!
- Arranging n objects where p are alike, q are alike: n! / (p! * q!)
- Circular permutations: (n-1)!
- Circular with no distinction of direction: (n-1)!/2

## Combinations (Order Doesn't Matter)
- nCr = n! / (r! * (n-r)!)
- nCr = nC(n-r)
- nC0 = nCn = 1
- nC1 = n
- nCr + nC(r-1) = (n+1)Cr (Pascal's rule)
- nC0 + nC1 + ... + nCn = 2^n

## Important Results
- Number of ways to distribute n identical items among r groups: (n+r-1)C(r-1)
- Derangements (no item in original position): D_n = n! * sum_{k=0}^{n} (-1)^k / k!
- Number of diagonals in n-gon: nC2 - n = n(n-3)/2

## Common Problem Types
- Selecting committees with constraints
- Arranging letters with repeated characters
- Distributing objects into groups
- Problems with "at least one" (use complement: total - none)
- Division into groups of equal size
