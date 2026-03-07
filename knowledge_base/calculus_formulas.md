# Calculus Formulas (Limits, Derivatives, Basic Optimization)

## Limits
- lim_{x->a} f(x) = L means f(x) approaches L as x approaches a
- lim_{x->0} sin(x)/x = 1
- lim_{x->0} (1 - cos(x))/x = 0
- lim_{x->0} (e^x - 1)/x = 1
- lim_{x->0} ln(1+x)/x = 1
- lim_{x->0} (a^x - 1)/x = ln(a)
- lim_{x->inf} (1 + 1/x)^x = e
- lim_{x->0} (1 + x)^(1/x) = e
- lim_{x->0} tan(x)/x = 1

## L'Hopital's Rule
- If lim f(x)/g(x) = 0/0 or inf/inf, then lim f(x)/g(x) = lim f'(x)/g'(x)
- Can be applied repeatedly if conditions persist

## Derivatives
### Basic Rules
- d/dx [c] = 0 (constant)
- d/dx [x^n] = n*x^(n-1) (power rule)
- d/dx [f+g] = f' + g' (sum rule)
- d/dx [f*g] = f'g + fg' (product rule)
- d/dx [f/g] = (f'g - fg') / g^2 (quotient rule)
- d/dx [f(g(x))] = f'(g(x)) * g'(x) (chain rule)

### Standard Derivatives
- d/dx [sin x] = cos x
- d/dx [cos x] = -sin x
- d/dx [tan x] = sec^2 x
- d/dx [e^x] = e^x
- d/dx [ln x] = 1/x
- d/dx [a^x] = a^x * ln(a)
- d/dx [sin^(-1) x] = 1/sqrt(1-x^2)
- d/dx [cos^(-1) x] = -1/sqrt(1-x^2)
- d/dx [tan^(-1) x] = 1/(1+x^2)

## Integration (Basic)
- integral x^n dx = x^(n+1)/(n+1) + C, n != -1
- integral 1/x dx = ln|x| + C
- integral e^x dx = e^x + C
- integral sin x dx = -cos x + C
- integral cos x dx = sin x + C
- integral sec^2 x dx = tan x + C

## Optimization
- Find critical points: set f'(x) = 0
- Second derivative test:
  - f''(c) > 0: local minimum at c
  - f''(c) < 0: local maximum at c
  - f''(c) = 0: inconclusive
- For closed interval [a,b]: check f at critical points and endpoints

## Applications
- Rate of change: dy/dx at a point
- Tangent line at (a, f(a)): y - f(a) = f'(a)(x - a)
- Normal line: y - f(a) = -1/f'(a) * (x - a)
- Maxima/minima word problems: set up function, differentiate, solve
