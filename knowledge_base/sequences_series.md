# Sequences and Series

## Arithmetic Progression (AP)
- Terms: a, a+d, a+2d, ..., a+(n-1)d
- nth term: a_n = a + (n-1)d
- Sum of n terms: S_n = n/2 * [2a + (n-1)d] = n/2 * (first + last)
- Arithmetic mean of a and b: AM = (a+b)/2
- If a, b, c are in AP: 2b = a + c

## Geometric Progression (GP)
- Terms: a, ar, ar^2, ..., ar^(n-1)
- nth term: a_n = a * r^(n-1)
- Sum of n terms: S_n = a(r^n - 1)/(r - 1), r != 1
- Sum to infinity (|r| < 1): S_inf = a/(1 - r)
- Geometric mean of a and b: GM = sqrt(ab)
- If a, b, c are in GP: b^2 = ac

## Harmonic Progression (HP)
- Reciprocals form an AP
- Harmonic mean of a and b: HM = 2ab/(a+b)
- AM >= GM >= HM (for positive reals)
- AM * HM = GM^2

## Special Series
- Sum of first n naturals: n(n+1)/2
- Sum of squares: n(n+1)(2n+1)/6
- Sum of cubes: [n(n+1)/2]^2

## Telescoping Series
- If a_k = f(k) - f(k+1), then sum = f(1) - f(n+1)
- Look for partial fractions that cancel

## Convergence Tests
- Geometric series converges iff |r| < 1
- Comparison test, ratio test for general series
