# Assignment-12
# ==========================================
# CHC-101: Assignment 12 - Weather Prediction
# ==========================================

# Import required libraries
from sympy import symbols, Function, Eq, dsolve, exp, cos, pi, fourier_transform, Matrix

# ==========================================
# 1. Temperature Relaxation Model
# ==========================================

# Define symbols and function
t, k, Ta = symbols('t k Ta')
T = Function('T')(t)

# Differential equation: dT/dt = -k(T - Ta)
ode = Eq(T.diff(t), -k*(T - Ta))

# Solve ODE symbolically with T(0) = 35
sol_T = dsolve(ode, ics={T.subs(t, 0): 35})
print("Solution for Temperature Relaxation Model:")
print(sol_T)
print()

# Evaluate T(t) for k=0.1, t=5
T_eval = sol_T.subs({k:0.1, t:5})
print("T(t) evaluated at k=0.1, t=5:")
print(T_eval)
print()

# Interpretation comments:
# As k -> 0, T(t) ~ 35 (slow cooling)
# As k -> ∞, T(t) -> Ta (instant cooling)


# ==========================================
# 2. Atmospheric Wave Analysis
# ==========================================

# Define variables
x, f = symbols('x f')

# Define p(x)
p = exp(-x**2/10) * cos(2*pi*x)

# Compute Fourier Transform
F = fourier_transform(p, x, f)
print("Fourier Transform of p(x):")
print(F)
print()

# Interpretation:
# The result shows two Gaussian peaks centered at f = ±1,
# indicating dominant frequencies at ±1 Hz.


# ==========================================
# 3. Weather Station Network
# ==========================================

# Define matrices
x = symbols('x')
A = Matrix([[30,101,60],
            [32,100,55],
            [29,102,65]])

B = Matrix([[0.6,0.2,0.2],
            [0.3,x,0.2],
            [0.1,0.3,0.6]])

# Compute C = A * B
C = A * B
print("Matrix C = A * B:")
print(C)
print()

# Eigenvalues of B
eigvals_B = B.eigenvals()
print("Eigenvalues of B:")
print(eigvals_B)
print()

# Determinant of B
det_B = B.det()
print("Determinant of B:")
print(det_B)
print()

# ==========================================
# End of Assignment
# ==========================================
