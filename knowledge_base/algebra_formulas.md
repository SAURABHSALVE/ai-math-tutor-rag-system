# Algebra Formulas and Identities

## Quadratic Equations
- Standard form: ax^2 + bx + c = 0
- Quadratic formula: x = (-b +/- sqrt(b^2 - 4ac)) / (2a)
- Discriminant D = b^2 - 4ac
  - D > 0: two distinct real roots
  - D = 0: two equal real roots (repeated root)
  - D < 0: two complex conjugate roots
- Sum of roots: -b/a
- Product of roots: c/a
- Vieta's formulas extend to higher degree polynomials

## Algebraic Identities
- (a + b)^2 = a^2 + 2ab + b^2
- (a - b)^2 = a^2 - 2ab + b^2
- a^2 - b^2 = (a + b)(a - b)
- (a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3
- (a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3
- a^3 + b^3 = (a + b)(a^2 - ab + b^2)
- a^3 - b^3 = (a - b)(a^2 + ab + b^2)

## Arithmetic and Geometric Progressions
- AP: a, a+d, a+2d, ...
  - nth term: a_n = a + (n-1)d
  - Sum of n terms: S_n = n/2 * (2a + (n-1)d)
- GP: a, ar, ar^2, ...
  - nth term: a_n = a * r^(n-1)
  - Sum of n terms: S_n = a(r^n - 1)/(r - 1), r != 1
  - Sum to infinity (|r| < 1): S = a/(1 - r)

## Logarithms
- log_a(xy) = log_a(x) + log_a(y)
- log_a(x/y) = log_a(x) - log_a(y)
- log_a(x^n) = n * log_a(x)
- Change of base: log_a(x) = log_b(x) / log_b(a)
- log_a(a) = 1, log_a(1) = 0

## Binomial Theorem
- (x + y)^n = sum_{k=0}^{n} C(n,k) * x^(n-k) * y^k
- C(n,k) = n! / (k!(n-k)!)
- General term T_{r+1} = C(n,r) * x^(n-r) * y^r

## Inequalities
- AM >= GM >= HM for positive reals
- AM = (a+b)/2, GM = sqrt(ab), HM = 2ab/(a+b)
- Cauchy-Schwarz: (sum a_i*b_i)^2 <= (sum a_i^2)(sum b_i^2)
- Triangle inequality: |a + b| <= |a| + |b|

## Polynomials
- Factor theorem: (x - a) is a factor of P(x) if P(a) = 0
- Remainder theorem: P(x) divided by (x - a) gives remainder P(a)
- Fundamental theorem of algebra: degree n polynomial has exactly n roots (counting multiplicity, over complex numbers)
