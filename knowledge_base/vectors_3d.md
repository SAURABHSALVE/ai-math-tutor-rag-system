# Vectors and 3D Geometry

## Vector Basics
- Position vector of point P(x,y,z): OP = xi + yj + zk
- Magnitude: |a| = sqrt(a1^2 + a2^2 + a3^2)
- Unit vector: a_hat = a / |a|
- Direction cosines: cos(alpha) = a1/|a|, cos(beta) = a2/|a|, cos(gamma) = a3/|a|
- cos^2(alpha) + cos^2(beta) + cos^2(gamma) = 1

## Vector Operations
- Addition: (a1+b1, a2+b2, a3+b3)
- Scalar multiplication: k(a1, a2, a3) = (ka1, ka2, ka3)

## Dot Product
- a . b = a1*b1 + a2*b2 + a3*b3 = |a||b|cos(theta)
- a . b = 0 iff a perpendicular to b
- Projection of a on b: (a . b) / |b|
- Component of a along b: (a . b / |b|^2) * b

## Cross Product
- a x b = |i  j  k; a1 a2 a3; b1 b2 b3|
- |a x b| = |a||b|sin(theta) = area of parallelogram
- a x b = 0 iff a parallel to b
- a x b = -(b x a) (anti-commutative)

## Scalar Triple Product
- [a b c] = a . (b x c) = det([a; b; c])
- |[a b c]| = volume of parallelepiped
- Coplanar iff [a b c] = 0

## Lines in 3D
- Vector form: r = a + t*b (a = point, b = direction)
- Cartesian: (x-x1)/l = (y-y1)/m = (z-z1)/n

## Planes
- General: ax + by + cz = d
- Normal vector: n = (a, b, c)
- Distance from point P to plane: |aP1 + bP2 + cP3 - d| / sqrt(a^2+b^2+c^2)
