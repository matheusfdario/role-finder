import numpy as np

def computeIntersectionBetweenCircles(R, r, h, k):
    # Círculo 1: Centro em (0, 0) e raio R;
    # Circulo 2: Centro em (h, k) e raio r;
    x = (h ** 3 +
         np.sqrt(-h ** 4 * k ** 2 - 2 * h ** 2 * k ** 4 + 2 * h ** 2 * k ** 2 * r ** 2 + 2 * h ** 2 * k ** 2 * R ** 2 -
                 k ** 6 + 2 * k ** 4 * r ** 2 + 2 * k ** 4 * R ** 2 - k ** 2 * r ** 4 + 2 * k ** 2 * r ** 2 * R ** 2 - k ** 2 * R ** 4)
         + h * k ** 2 - h * r ** 2 + h * R ** 2) / (2 * (h ** 2 + k ** 2))
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


def circle_equation(radius):
    # Metade superior do círculo
    return lambda x: np.sqrt(radius ** 2 - x ** 2)


def ellipse_equation(a, b, c):
    # Metade superior da elipse
    return lambda x: np.sqrt((b ** 2) * (1 - ((x - c) ** 2) / (a ** 2)))
