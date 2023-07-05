import numpy as np
import scipy.optimize as optm



def computeIntersectionBetweenCircles(R, r, h, k):
    # Círculo 1: Centro em (0, 0) e raio R;
    # Circulo 2: Centro em (h, k) e raio r;
    x = (h**3 +
         np.sqrt(-h**4 * k**2 - 2 * h**2 * k**4 + 2 * h**2 * k**2 * r**2 + 2 * h**2 * k**2 * R**2 -
                 k**6 + 2 * k**4 * r**2 + 2 * k**4 * R**2 - k**2 * r**4 + 2 * k**2 * r**2 * R**2 - k**2 * R**4)
         + h * k**2 - h * r**2 + h * R**2)/(2 * (h**2 + k**2))
    return x

def findCircle(A, B, C):
    # Encontra o círculo de raio R e centro em (P[0], P[1]) que toca nos pontos A, B e C;
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


def center_part_curve_scalar(x, v1, v2, time_bda, B):
    f = (-(2*v2**4 * x * B[0] - v1**4 * x**2 - v2**4 * B[0]**2 - v2**4 * x**2 + time_bda**2 * v2**2 * v1**4
           + time_bda**2 * v2**4 * v1**2 + 2 * v2**2 * v1**2 * x**2 + v2**2 * v1**2 * (B[0])**2
           - 2 * time_bda * v2**2 * v1**2 *
           (time_bda**2 * v2**2 * v1**2 - v2**2 * (B[0])**2
            + 2 * x * v2**2 * B[0] + v1**2 * (B[0])**2 - 2 * x * v1**2 * B[0])**(1 / 2)
           - 2 * v2**2 * v1**2 * x * B[0])**(1 / 2))\
        / (v2**2 - v1**2)
    return f

def center_part_curve(x, v1, v2, time_bda, B):
    fcn = lambda x : center_part_curve_scalar(x, v1, v2, time_bda, B)
    fnc_vectorized = np.frompyfunc(fcn, nin=1, nout=1)
    return fnc_vectorized(x)

def line_func(x, beta, offset):
    return (x - offset)*np.tan(beta)

def center_area_intersection(circle_angle, c, r_center, v1, v2, time_bda, A, B):
    beta = np.deg2rad(circle_angle)
    ac = np.arcsin(v2 / v1)

    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])

    r, h, k = findCircle(B, A, En)
    xint = computeIntersectionBetweenCircles(r_center, r, h, k)

    curve = lambda x : center_part_curve(x, v1, v2, time_bda, B)
    phi = lambda x: np.power(line_func(x, beta, 0) - curve(x), 2)
    xi = optm.minimize_scalar(phi, bounds=(xint, 85.24284999101062), method='bounded')
    xi = xi.x
    yi = curve(xi)

    return (xi, yi)


def side_area_intersection(circle_angle, a, c, A):
    beta = np.deg2rad(circle_angle)
    A = 2 * c

    sol = optm.root(lambda B: - (B**2) + A**2 + (2*a - B)**2 - 2*A*(2*a - B) * np.cos(beta), 0)
    B = sol.x[0]
    C = 2 * a - B

    xi = C * np.cos(beta)
    yi = C * np.sin(beta)

    return xi, yi

def compute_entrypoints(circle_angle, a, c, r_center, v1, v2, A, B, N):
    ac = np.arcsin(v2 / v1)

    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])
    r, h, k = findCircle(B, A, En)

    xint = computeIntersectionBetweenCircles(r_center, r, h, k)
    yint = np.sqrt(r_center ** 2 - xint ** 2)

    ad = np.sqrt((A[0] - xint) ** 2 + (A[1] - yint) ** 2)
    db = np.sqrt((B[0] - xint) ** 2 + (B[1] - yint) ** 2)

    time_ad = ad / v2
    time_db = db / v1
    time_bda = time_db + time_ad

    angle_a = 0
    angle_b = np.rad2deg(np.arctan(N[1]/N[0]))
    angle_c = 90

    beta = np.abs(circle_angle)

    if angle_a <= beta <= angle_b:
        xi, yi = center_area_intersection(beta, c, r_center, v1, v2, time_bda, A, B)
    elif angle_b < beta <= angle_c:
        xi, yi = side_area_intersection(beta, a, c, A)
    else:
        print("Ângulo não suportado")

    if circle_angle < 0:
        yi = -yi

    alpha = np.float(np.arctan(yi/(2*c - xi)))

    return alpha, (xi, yi)