# Application of Derivatives

## Rate of Change
- dy/dx represents the rate of change of y with respect to x
- If s(t) is position: v(t) = ds/dt (velocity), a(t) = dv/dt (acceleration)

## Tangent and Normal Lines
- Slope of tangent at (a, f(a)): m = f'(a)
- Equation of tangent: y - f(a) = f'(a)(x - a)
- Slope of normal: -1/f'(a)
- Equation of normal: y - f(a) = -1/f'(a) * (x - a)

## Increasing / Decreasing Functions
- f is increasing on interval if f'(x) > 0
- f is decreasing on interval if f'(x) < 0
- f is constant if f'(x) = 0

## Maxima and Minima
### First Derivative Test
- f'(x) changes from + to - at c: local maximum
- f'(x) changes from - to + at c: local minimum
- No sign change: neither

### Second Derivative Test
- If f'(c) = 0 and f''(c) > 0: local minimum at c
- If f'(c) = 0 and f''(c) < 0: local maximum at c
- If f''(c) = 0: test is inconclusive

### Global Extrema on [a,b]
- Find all critical points in (a,b)
- Evaluate f at critical points and endpoints a, b
- Largest value is global max, smallest is global min

## Rolle's Theorem
- If f is continuous on [a,b], differentiable on (a,b), and f(a) = f(b)
- Then there exists c in (a,b) such that f'(c) = 0

## Mean Value Theorem (MVT)
- If f is continuous on [a,b] and differentiable on (a,b)
- There exists c in (a,b) such that f'(c) = (f(b) - f(a))/(b - a)
