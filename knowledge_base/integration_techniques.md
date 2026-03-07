# Integration Techniques

## Basic Integrals
- integral x^n dx = x^(n+1)/(n+1) + C, n != -1
- integral 1/x dx = ln|x| + C
- integral e^x dx = e^x + C
- integral a^x dx = a^x / ln(a) + C
- integral sin x dx = -cos x + C
- integral cos x dx = sin x + C
- integral sec^2 x dx = tan x + C
- integral csc^2 x dx = -cot x + C
- integral sec x tan x dx = sec x + C
- integral 1/(1+x^2) dx = tan^(-1)(x) + C
- integral 1/sqrt(1-x^2) dx = sin^(-1)(x) + C

## Integration by Substitution
- If integral f(g(x)) g'(x) dx, let u = g(x)
- Then du = g'(x) dx, integral becomes integral f(u) du

## Integration by Parts
- integral u dv = uv - integral v du
- LIATE rule for choosing u: Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential

## Partial Fractions
- For rational functions P(x)/Q(x) where deg(P) < deg(Q)
- Linear factor (ax+b): A/(ax+b)
- Repeated linear: A/(ax+b) + B/(ax+b)^2
- Quadratic factor: (Ax+B)/(ax^2+bx+c)

## Definite Integrals Properties
- integral_a^b f(x) dx = -integral_b^a f(x) dx
- integral_a^b f(x) dx = integral_a^c f(x) dx + integral_c^b f(x) dx
- integral_0^a f(x) dx = integral_0^a f(a-x) dx
- If f is even: integral_{-a}^a f(x) dx = 2 * integral_0^a f(x) dx
- If f is odd: integral_{-a}^a f(x) dx = 0
