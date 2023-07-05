import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optm
from matplotlib.patches import Ellipse

import scipy.optimize as optm

from generate_law import generate_law
from smartwedge import *

if __name__ == "__main__":
    # Arco que liga elipse a região plana do transdutor:
    # rarc = np.sqrt(9510.59)
    # x_center = 71.15
    # y_center = 24.38

    # Ellipse parameters:
    c = 84.28
    r0 = 67.15
    wc = 6.2
    offset = 2
    Tprime = 19.5


    # Velocidade do som nos materiais em km/s
    v1 = 6.37
    v2 = 1.43


    # Localizaçã dos dois focos da elípse:
    A = np.array([0, 0])
    B = np.array([2 * c, 0])


    ac = np.arcsin(v2 / v1)
    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])
    r, h, k = findCircle(B, A, En)
    R = r0 + wc


    xint = computeIntersectionBetweenCircles(r0 + wc, r, h, k)
    yint = np.sqrt(R ** 2 - xint ** 2)

    ad = np.sqrt((A[0] - xint) ** 2 + (A[1] - yint) ** 2)
    db = np.sqrt((B[0] - xint) ** 2 + (B[1] - yint) ** 2)

    time_ad = ad / v2
    time_db = db / v1
    time_bda = time_db + time_ad

    curve = lambda x: center_part_curve(x, v1, v2, time_bda, B)
    # Encontra qual é a reta que é tangente a circunferência de raio r0+wc e toca no ponto Tprime:
    # Equação da CircunferÊncia centrada em (0,0); f(x) = np.sqrt(r**2 - x**2)
    # A derivada será: df(x)dx = -x/(np.sqrt(r**2 - x**2)
    # Onde é válido apenas no intervalo: [-r, r]
    # A reta então de mesma inclinação deverá ser:
    # y(x) = dfdx * x + b
    # Para o caso em que x=2*c => y(x) = Tprime
    # Tprime = an * (2*c) + b
    N = np.array([53.26, curve([53.26])], dtype=float)  # Ponto arbitrário na curva paramétrica;
    raio = np.linalg.norm(N)
    f_circle = lambda x: np.sqrt(raio ** 2 - x ** 2)
    dfdx_circle = lambda x: -x / (np.sqrt(raio ** 2 - x ** 2))
    b1 = lambda xt: - Tprime + (xt * (2 * c)) / (np.sqrt(raio ** 2 - xt ** 2))
    b2 = lambda xt: f_circle(xt) - xt * dfdx_circle(xt)

    cost_fun = lambda x: np.power(b1(x) - b2(x), 2)
    xT = optm.minimize_scalar(cost_fun, bounds=(0, r0 + wc), method='bounded').x
    yT = f_circle(xT)
    an = dfdx_circle(xT)
    bn = -Tprime - an * (2 * c)
    tangent_line_circle = lambda x: an * x + bn
    Fs = np.array([0, tangent_line_circle(0) + offset])

    # Descobrir equação da elipse que possui como foco A e B e passa pelo ponto (0, Fs):

    # a =np.sqrt( 20143.2)
    # b = np.sqrt(13040.93)

    b = lambda a : np.sqrt(a**2 - c**2)
    ellipse_costfun = lambda a : np.power(((Fs[0] - c)**2)/(a**2) + (Fs[1]**2)/(b(a)**2) - 1, 2)
    a = optm.minimize_scalar(ellipse_costfun, bounds=(10, 300), method='bounded').x
    b = np.sqrt(a**2 - c**2)

    z = np.arange(0, 2 * c)
    z_circle = np.arange(0, r0 + wc)
    plt.plot(z_circle, f_circle(z_circle))
    plt.scatter(xT, yT, color='red')
    plt.plot(z, tangent_line_circle(z))
    cylinder_circle = plt.Circle((0, 0), r0 + wc, facecolor='None', edgecolor='k')
    ax = plt.gca()
    ax.add_patch(cylinder_circle)
    plt.axis('equal')

    plt.figure()
    step = 5
    betas = np.arange(-90, 90+step, step)

    # Coordendada máxima para v1 = 6.7 mm/us e v2 = 1.43 mm/us é 84.54793576226876
    x = np.arange(xint, 85.08749525101061, 1e-5)

    ax = plt.gca()

    x_offset = a - c
    center_circle = plt.Circle((h, k), r, facecolor='None', edgecolor='k', linestyle=":")
    specimen_circle = plt.Circle((0, 0), r0-20, facecolor='None', edgecolor='r')
    cylinder_circle = plt.Circle((0, 0), raio, facecolor='None', edgecolor='k')
    ellipse_externo = Ellipse((a - x_offset, 0), 2 * a, 2 * b, facecolor='None', edgecolor='g')

    plt.scatter(*A, label='Centro do Espécime')
    plt.scatter(*B, label='Centro do Transdutor')
    t = np.arange(xint, 85.08749525101061, 1e-3)

    plt.plot(t, curve(t), label='Superfície Parametrizada')
    plt.plot(t, -curve(t), label='_')

    planewave_angle = np.zeros_like(betas, dtype=float)
    k = 0

    for k, beta in enumerate(betas):
        alpha, intersect = compute_entrypoints(beta, a, c, r0 + wc, v1, v2, A, B, N)
        planewave_angle[k] = alpha
        t1 = np.arange(0, intersect[0], 1e-3)
        t2 = np.arange(intersect[0], 2*c, 1e-3)
        plt.scatter(*intersect, s=.7, color=[1, 0, 1])

        xmed = intersect[0]
        ymed = intersect[1]

        xvec = [0, xmed, 2*c]


        yvec = [0, ymed, 0]
        plt.plot(xvec, yvec, ':', color=[1, 0, 0])
        plt.scatter(xvec[1], yvec[1], color=[0, 0, 0])

    angle_a = 0
    angle_b = np.rad2deg(np.arctan(N[1] / N[0]))
    angle_c = 90


    ax.add_patch(center_circle)
    ax.add_patch(cylinder_circle)
    ax.add_patch(ellipse_externo)
    ax.add_patch(specimen_circle)
    plt.axis('equal')
    plt.legend(loc='lower right')

    plt.figure()
    plt.plot(betas, np.rad2deg(planewave_angle), 'o')
    plt.xlabel("Ângulo de Varredura do Círculo em graus")
    plt.ylabel("Ângulo da Planewave em graus")
    plt.xlim([-95, 95])
    plt.xticks(np.linspace(-95, 95, 9))
    plt.grid()

    # planewave_angle = np.rad2deg(planewave_angle)
    # generate_law_sectorial(data.probe_params.elem_center[:, 0], planewave_angle, data.specimen_params.cl)

