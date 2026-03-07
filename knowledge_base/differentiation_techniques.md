# Differentiation Techniques

## Basic Rules
- d/dx [constant] = 0
- d/dx [x^n] = n * x^(n-1) (power rule)
- d/dx [f + g] = f' + g' (sum rule)
- d/dx [c * f] = c * f' (constant multiple)
- d/dx [f * g] = f'g + fg' (product rule)
- d/dx [f/g] = (f'g - fg')/g^2 (quotient rule)
- d/dx [f(g(x))] = f'(g(x)) * g'(x) (chain rule)

## Implicit Differentiation
- When y is defined implicitly by F(x,y) = 0
- Differentiate both sides with respect to x
- Treat y as a function of x: d/dx[y] = dy/dx
- Solve for dy/dx

## Logarithmic Differentiation
- Take ln of both sides: ln(y) = ln(f(x))
- Differentiate: (1/y)(dy/dx) = d/dx[ln(f(x))]
- Useful for y = f(x)^g(x) type problems

## Parametric Differentiation
- If x = f(t), y = g(t):
- dy/dx = (dy/dt) / (dx/dt)
- d^2y/dx^2 = (d/dt)(dy/dx) / (dx/dt)

## Higher Order Derivatives
- f''(x) = d/dx[f'(x)]
- Leibniz formula: (fg)^(n) = sum_{k=0}^{n} nCk * f^(k) * g^(n-k)

## Derivatives of Special Functions
- d/dx[|x|] = x/|x| = sgn(x), x != 0
- d/dx[ln|x|] = 1/x
- d/dx[e^(f(x))] = e^(f(x)) * f'(x)
- d/dx[a^(f(x))] = a^(f(x)) * ln(a) * f'(x)
