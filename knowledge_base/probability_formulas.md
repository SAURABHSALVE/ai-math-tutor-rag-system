# Probability Formulas and Concepts

## Basic Probability
- P(A) = favorable outcomes / total outcomes
- 0 <= P(A) <= 1
- P(A') = 1 - P(A) (complement)
- P(A union B) = P(A) + P(B) - P(A intersect B)
- P(A intersect B) = P(A) * P(B|A)

## Conditional Probability
- P(A|B) = P(A intersect B) / P(B)
- Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
- Total Probability: P(B) = sum P(B|A_i) * P(A_i) for partition {A_i}

## Independent Events
- A and B independent iff P(A intersect B) = P(A) * P(B)
- If independent: P(A|B) = P(A)

## Permutations and Combinations
- nPr = n! / (n-r)! (ordered arrangements)
- nCr = n! / (r!(n-r)!) (unordered selections)
- nCr = nC(n-r)
- nC0 + nC1 + ... + nCn = 2^n

## Distributions
### Binomial Distribution
- P(X = k) = C(n,k) * p^k * (1-p)^(n-k)
- Mean = np, Variance = np(1-p)

### Poisson Distribution
- P(X = k) = (lambda^k * e^(-lambda)) / k!
- Mean = Variance = lambda

### Normal Distribution
- PDF: f(x) = (1/(sigma*sqrt(2*pi))) * e^(-(x-mu)^2 / (2*sigma^2))
- 68-95-99.7 rule for standard deviations

## Expected Value and Variance
- E(X) = sum x_i * P(X = x_i)
- Var(X) = E(X^2) - [E(X)]^2
- E(aX + b) = aE(X) + b
- Var(aX + b) = a^2 * Var(X)

## Common JEE Problem Types
- Drawing balls from urns (with/without replacement)
- Dice rolling problems
- Card selection problems
- Conditional probability with Bayes' theorem
- Geometric probability (area-based)
