import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optm
from matplotlib.patches import Ellipse

from generate_law import generate_law
from smartwedge import *

if __name__ == "__main__":
    # Ellipse parameters:
    a = 141.868
    b = 114.124
    c = 82.275
    r0 = 67.15
    wc = 6.2

    # Localizaçã dos dois focos da elípse:
    A = np.array([0, 0])
    B = np.array([2 * c, 0])

    # Velocidade do som nos materiais em km/s
    v1 = 6.37
    v2 = 1.43


    ac = np.arcsin(v2 / v1)
    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])
    r, h, k = findCircle(B, A, En)
    R = r0 + wc


    xint = computeIntersectionBetweenCircles(r0 + wc, r, h, k)
    yint = np.sqrt(R ** 2 - xint ** 2)



    angle_a = 0
    angle_b = np.rad2deg(np.arctan(yint / xint))
    angle_c = 90


    # Região de incidência indireta:
    vetor1 = list()
    for beta in np.arange(angle_b, angle_c):
        alpha, intersect = compute_entrypoints(beta, a, c, r0 + wc, v1, v2, A, B)
        t_bd = np.linalg.norm(intersect - B)/(v1 * 1e3) * 1e-3
        t_dw = (np.linalg.norm(A - intersect) - (r0 + wc))/(v1 * 1e3) * 1e-3
        t_ws = (wc * 1e-3)/(v2 * 1e3)
        t_sb = r0 / (5300)

        t_total_periferia = 2 * (t_bd + t_dw + t_ws + t_sb)
        vetor1.append(t_total_periferia)



    t1_mean = np.mean(vetor1)

    # Região de incidência direta:
    vetor2 = list()
    for beta in np.arange(angle_a, angle_b):
        alpha, intersect = compute_entrypoints(beta, a, c, r0 + wc, v1, v2, A, B)
        t_bw = np.linalg.norm(intersect - B) / (v1 * 1e3) * 1e-3
        t_ws = (np.linalg.norm(A - intersect) - r0) / (v2 * 1e3) * 1e-3
        t_sa = (r0) / (5300) * 1e-3
        t_total_centro = 2 * (t_bw + t_ws + t_sa)
        vetor2.append(t_total_centro)


    t2_mean = np.mean(vetor2)


