# Limits and Continuity

## Definition of Limit
- lim_{x->a} f(x) = L if for every epsilon > 0 there exists delta > 0 such that |f(x) - L| < epsilon whenever 0 < |x - a| < delta

## Standard Limits
- lim_{x->0} sin(x)/x = 1
- lim_{x->0} tan(x)/x = 1
- lim_{x->0} (e^x - 1)/x = 1
- lim_{x->0} (a^x - 1)/x = ln(a)
- lim_{x->0} ln(1+x)/x = 1
- lim_{x->0} (1+x)^(1/x) = e
- lim_{x->inf} (1 + 1/x)^x = e
- lim_{x->0} (1 - cos x)/x^2 = 1/2
- lim_{x->0} sin^(-1)(x)/x = 1
- lim_{x->0} tan^(-1)(x)/x = 1

## Indeterminate Forms
- 0/0, inf/inf, 0 * inf, inf - inf, 0^0, 1^inf, inf^0
- All can be converted to 0/0 or inf/inf form for L'Hopital

## L'Hopital's Rule
- If lim f(x)/g(x) gives 0/0 or inf/inf:
- lim f(x)/g(x) = lim f'(x)/g'(x) (if the right side exists)
- Can apply repeatedly

## Sandwich / Squeeze Theorem
- If g(x) <= f(x) <= h(x) and lim g(x) = lim h(x) = L, then lim f(x) = L

## Continuity
- f is continuous at x = a if: lim_{x->a} f(x) = f(a)
- Requirements: f(a) exists, limit exists, they are equal
- Polynomials are continuous everywhere
- Rational functions are continuous except where denominator = 0

## Types of Discontinuity
- Removable: limit exists but != f(a) or f(a) undefined
- Jump: left limit != right limit
- Infinite: function goes to +/- infinity
