import numpy as np
import matplotlib.pyplot as plt
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

def computeIntersection(R, r, h, k):
    x = (h**3 + np.sqrt(-h**4 * k**2 - 2 * h**2 * k**4 + 2 * h**2 * k**2 * r**2 + 2 * h**2 * k**2 * R**2 - k**6 + 2 * k**4 * r**2 + 2 * k**4 * R**2 - k**2 * r**4 + 2 * k**2 * r**2 * R**2 - k**2 * R**4) + h * k**2 - h * r**2 + h * R**2)/(2 * (h**2 + k**2))
    return x

def line_origin(x, beta):
    beta = np.deg2rad(beta)
    return x*np.tan(beta)

def curva(X, v1, v2, time_bda, B):
    output = np.zeros_like(X)
    for i, x in enumerate(X):
        f = (-(2*v2**4 * x * B[0] - v1**4 * x**2 - v2**4 * B[0]**2 - v2**4 * x**2 + time_bda**2 * v2**2 * v1**4 + time_bda**2 * v2**4 * v1**2 + 2 * v2**2 * v1**2 * x**2 + v2**2 * v1**2 * (B[0])**2 - 2* time_bda * v2**2 * v1**2 * (time_bda**2 * v2**2 * v1**2 - v2**2 * (B[0])**2 + 2 * x * v2**2 * B[0] + v1**2 * (B[0])**2 - 2 * x * v1**2 * B[0])**(1 / 2) - 2 * v2**2 * v1**2 * x * B[0])**(1 / 2)) / (v2**2 - v1**2)
        output[i] = f
    return output

def ellipse(x, a, b, c):
    focus_x = c
    y = np.sqrt((b**2) * (1 - ((x - focus_x)**2)/(a**2)))
    return y

def circle_func(x, r, h, k):
    return np.sqrt(r**2 - (x-h)**2) + k



a = 141.868
b = 114.124
c = 82.275
r0 = 67.15
wc = 6.2

r_specimen = np.sqrt(5677.34)
t1 = np.arange(-(a-c), 2*a - (a-c), 1e-3)
t2 = np.arange(-r_specimen, r_specimen, 1e-3)
yt1 = ellipse(t1, a, b, c)
yt2 = circle_func(t2, r_specimen, 0, 0)

phi = lambda x : np.power(ellipse(x, a, b, c) - circle_func(x, r_specimen, 0, 0), 2)
xintersect = minimize_scalar(phi, bounds=(t2[0], t2[-1]), method='bounded').x
yintersect = circle_func(xintersect, r_specimen, 0, 0)

plt.scatter(xintersect, yintersect)

plt.plot(t1, yt1)
plt.plot(t2, yt2)
plt.axis('equal')

def find_planewave_angle(circle_angle):
    beta = np.deg2rad(circle_angle)

    # Parâmetros da sapata e specimen:
    a = 141.868
    b = 114.124
    c = 82.275
    r0 = 67.15
    wc = 6.2

    # Parte relativa à reflexão na superfície elíptice:

    beta_max = np.pi/2

    A = np.array([0, 0])
    B = np.array([2 * c, 0])

    v1 = 6.7
    v2 = 1.43
    ac = np.arcsin(v2 / v1)

    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])

    r, h, k = findCircle(B, A, En)
    R = r0 + wc

    xint = computeIntersection(r0 + wc, r, h, k)
    yint = np.sqrt(R ** 2 - xint ** 2)

    beta_min = np.arctan(yint/xint)



    # Intersecção entre elipse de raio da região de reflexão:

    if beta <= beta_max and beta >= beta_min:
        1
    else:
        ad = np.sqrt((A[0] - xint) ** 2 + (A[1] - yint) ** 2)
        db = np.sqrt((B[0] - xint) ** 2 + (B[1] - yint) ** 2)

        time_ad = ad / v2
        time_db = db / v1
        time_bda = time_db + time_ad
        x = np.arange(xint, 84.54793576226876, 1e-5)
        phi = lambda var: np.power(line_origin(var, np.rad2deg(beta)) - curva([var], v1, v2, time_bda, B), 2)
        xi = minimize_scalar(phi, bounds=(xint, 84.54793576226876), method='bounded')
        xi = xi.x
        yi = curva([xi], v1, v2, time_bda, B)[0]
        intesection = (xi, yi)
    return intesection






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


beta = 4
intersect = find_planewave_angle(beta)
xi, yi = intersect

c = np.arange(0, xi, 1e-5)
plt.scatter(*intersect)
plt.plot(c, line_origin(c, beta))



ax = plt.gca()

center_circle = plt.Circle((h, k), r, facecolor='None', edgecolor='k', linestyle=":")
cylinder_circle = plt.Circle((0, 0), r0 + wc, facecolor='None', edgecolor='k')

ax.add_patch(center_circle)
ax.add_patch(cylinder_circle)
plt.axis('equal')
plt.legend(loc = 'lower right')