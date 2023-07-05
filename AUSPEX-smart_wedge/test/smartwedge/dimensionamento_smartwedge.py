import matplotlib
import scipy.optimize as optm
matplotlib.use('TkAgg')


import matplotlib.pyplot as plt
from prettytable import PrettyTable
from smartwedge.smartwedge import Smartwedge
from smartwedge.geometric_utils import *


def circle_equation(x, radius, h, k):
    y = np.sqrt(radius ** 2 - (x - h) ** 2) + k
    return y


if __name__ == "__main__":
    # Parêmtros da Elipse:
    #
    # c = 84.28
    # r0 = 67.15
    # wc = 6.2
    # offset = 2
    # Tprime = 19.5

    # Cria um objeto do tipo smartwedge_old:
    # Ellipse parameters:
    c = 84.28
    r0 = 70
    wc = 8
    offset = 2
    Tprime = 19.5

    # Velocidade do som nos materiais em mm/us
    v1 = 6.46
    v2 = 1.43
    v_steel = 5.9

    # Espessura da parede da tubulação em milímetros:
    tube_wall_width = 15

    # Criação do objeto smartwedge_old:
    sm = Smartwedge(c, r0, wc, v1, v2, Tprime, offset)

    # Localização dos dois focos da elípse:
    A = np.array([0, 0])
    B = np.array([2 * c, 0])

    # Ângulo crítico:
    ac = np.arcsin(v2 / v1)
    En = np.array([
        (A[0] + B[0]) / 2,
        c / (1 / np.cos(ac) + np.tan(ac))
    ])
    r, h, k = findCircle(B, A, En)

    xint = computeIntersectionBetweenCircles(r0 + wc, r, h, k)
    yint = np.sqrt((r0 + wc) ** 2 - xint ** 2)
    D = np.array([xint, yint])

    da = np.linalg.norm(A - D)
    bd = np.linalg.norm(B - D)

    time_ad = da / v2
    time_db = bd / v1
    time_bda = time_db + time_ad

    # Ângulos de leitura da circuferência:
    step = 1
    betas = np.arange(0, 90 + step, step)


    # Desenha a smartwedge_old:
    sm.draw(plot_tube=True, plot_control_points=True)

    plt.xlabel('Eixo x em milímetros')
    plt.ylabel('Eixo y em milímetros')
    plt.axis('equal')
    plt.ylim([-130, 130])
    plt.grid(alpha=0.5)

    plt.show()
