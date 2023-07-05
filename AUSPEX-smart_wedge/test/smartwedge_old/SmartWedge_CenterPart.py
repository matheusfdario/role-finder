import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import root, minimize_scalar


def findCircle(A, B, C):
    a = np.linalg.norm(C - B)
    b = np.linalg.norm(C - A)
    c = np.linalg.norm(B - A)
    s = (a + b + c) / 2
    R = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
    b1 = a * a * (b * b + c * c - a * a)
    b2 = b * b * (a * a + c * c - b * b)
    b3 = c * c * (a * a + b * b - c * c)
    P = np.column_stack((A, B, C)).dot(np.hstack((b1, b2, b3)))
    P /= b1 + b2 + b3

    return (R, P[0], P[1])

def computeIntersectionBetweenCircles(R, r, h, k):
    x = (h**3 + np.sqrt(-h**4 * k**2 - 2 * h**2 * k**4 + 2 * h**2 * k**2 * r**2 + 2 * h**2 * k**2 * R**2 - k**6 + 2 * k**4 * r**2 + 2 * k**4 * R**2 - k**2 * r**4 + 2 * k**2 * r**2 * R**2 - k**2 * R**4) + h * k**2 - h * r**2 + h * R**2)/(2 * (h**2 + k**2))
    return x

def line_func(x, beta, offset):
    beta = - beta
    return (x - offset)*np.tan(beta)

def curva(X, v1, v2, time_bda, B):
    output = np.zeros_like(X)
    for i, x in enumerate(X):
        f = (-(2*v2**4 * x * B[0] - v1**4 * x**2 - v2**4 * B[0]**2 - v2**4 * x**2 + time_bda**2 * v2**2 * v1**4 + time_bda**2 * v2**4 * v1**2 + 2 * v2**2 * v1**2 * x**2 + v2**2 * v1**2 * (B[0])**2 - 2* time_bda * v2**2 * v1**2 * (time_bda**2 * v2**2 * v1**2 - v2**2 * (B[0])**2 + 2 * x * v2**2 * B[0] + v1**2 * (B[0])**2 - 2 * x * v1**2 * B[0])**(1 / 2) - 2 * v2**2 * v1**2 * x * B[0])**(1 / 2)) / (v2**2 - v1**2)
        output[i] = f
    return output


# Ellipse parameters:
a = 141.868
b = 114.124
c = 82.275
r0 = 67.15
wc = 6.2


#
# alpha_min = 90 - np.rad2deg(np.arctan(53.69/52.86))
# alpha_max = 90 - np.rad2deg(np.arctan(91.81/(2*c)))
# alpha = 90 - np.arange(alpha_min, alpha_max, 1)
# alpha = np.deg2rad(alpha)

A = np.array([0, 0])
B = np.array([2*c, 0])

v1 = 6.7
v2 = 1.43
ac = np.arcsin(v2/v1)

En = np.array([
    (A[0] + B[0])/2,
    c/(1/np.cos(ac) + np.tan(ac))
])

r, h, k = findCircle(B, A, En)
R = r0 + wc

xint = computeIntersection(r0 + wc, r, h, k)
yint = np.sqrt(R**2 - xint**2)

ad = np.sqrt((A[0] - xint)**2 + (A[1] - yint)**2)
db = np.sqrt((B[0] - xint)**2 + (B[1] - yint)**2)

time_ad = ad / v2
time_db = db / v1
time_bda = time_db + time_ad



plt.figure()
plt.scatter(*A, label='Centro do Espécime')
plt.scatter(*B, label='Centro do Transdutor')
plt.scatter(*En)
plt.scatter(xint, yint)


t = np.arange(xint, 84.54793576226876, 1e-3)
ft = curva(t, v1, v2, time_bda, B)
plt.plot(t, ft, label='Superfície Parametrizada')


alpha = np.deg2rad(22)
x = np.arange(xint, 84.54793576226876, 1e-5)
phi = lambda var : np.power(line_func(var, alpha, 2*c) - curva([var], v1, v2, time_bda, B), 2)
xi = minimize_scalar(phi, bounds=(xint, 84.54793576226876), method='bounded')
xi = xi.x
yi = curva([xi], v1, v2, time_bda, B)[0]
intersect = (xi, yi)

t = np.arange(xi, 2*c, 1e-5)
plt.scatter(*intersect)
plt.plot(t, line_func(t, alpha, 2*c))



ax = plt.gca()

center_circle = plt.Circle((h, k), r, facecolor='None', edgecolor='k', linestyle=":")
cylinder_circle = plt.Circle((0, 0), r0 + wc, facecolor='None', edgecolor='k')

ax.add_patch(center_circle)
ax.add_patch(cylinder_circle)
plt.axis('equal')
plt.legend(loc = 'lower right')