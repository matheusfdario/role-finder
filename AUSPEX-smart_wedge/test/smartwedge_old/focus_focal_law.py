import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optm
from matplotlib.patches import Ellipse

import scipy.optimize as optm

from generate_law import generate_law
from smartwedge import *
from class_smartwedge import *
from generate_law import *


def find_focus_coord(r_focus, ang_focus_deg, smartwedge):
    # Encontra o ângulo e a distância do foco em relação ao referencial do transdutor.
    # r_focus é a distância do foco para o centro da tubulação
    # ang_focus_deg é posição angular do foco em relação ao centro da tubulação

    angle_a = 0
    angle_b = np.rad2deg(np.arctan(smartwedge.N[1] / smartwedge.N[0]))
    angle_c = 90

    radius = np.linalg.norm(smartwedge.N)
    # Checa se o foco está na direção da incidÊncia direta:
    if angle_a <= ang_focus_deg < angle_b:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        dist_bw = np.linalg.norm(intersect - smartwedge.B) * 1e-3 # Dist. entre centro do transdutor e incidência direta
        dist_ws = (np.linalg.norm(smartwedge.A - intersect) - smartwedge.r0) * 1e-3 # Dist. entre tubulação e sapata
        dist_sa = (smartwedge.r0 - r_focus) * 1e-3 # Distância entre tubulação e foco

        focus_dist = dist_bw + dist_ws + dist_sa

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        dist_bd = np.linalg.norm(intersect - smartwedge.B) * 1e-3 # Distância entre centro do transdutor e elipse
        dist_dw = (np.linalg.norm(smartwedge.A - intersect) - radius) * 1e-3 # Dist. entre elipse e borda da sapata
        dist_ws = (radius - smartwedge.r0) * 1e-3 # Distância entre sapata e tubulação
        dist_sb = (smartwedge.r0 - r_focus) * 1e-3 # Distância entre tubulação e foco

        focus_dist = dist_bd + dist_dw + dist_ws + dist_sb


    return focus_dist, np.rad2deg(focus_angle)



if __name__ == "__main__":
    elem_coord_x = np.array([-0.01575, -0.01525, -0.01475, -0.01425, -0.01375, -0.01325,
       -0.01275, -0.01225, -0.01175, -0.01125, -0.01075, -0.01025,
       -0.00975, -0.00925, -0.00875, -0.00825, -0.00775, -0.00725,
       -0.00675, -0.00625, -0.00575, -0.00525, -0.00475, -0.00425,
       -0.00375, -0.00325, -0.00275, -0.00225, -0.00175, -0.00125,
       -0.00075, -0.00025,  0.00025,  0.00075,  0.00125,  0.00175,
        0.00225,  0.00275,  0.00325,  0.00375,  0.00425,  0.00475,
        0.00525,  0.00575,  0.00625,  0.00675,  0.00725,  0.00775,
        0.00825,  0.00875,  0.00925,  0.00975,  0.01025,  0.01075,
        0.01125,  0.01175,  0.01225,  0.01275,  0.01325,  0.01375,
        0.01425,  0.01475,  0.01525,  0.01575]) # Posição do centro de cada elemento do transdutor em metros;
    elem_coord = np.zeros((elem_coord_x.shape[0], 2))
    elem_coord[:, 0] = elem_coord_x

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

    # Velocidade do som nos materiais em mm/us
    v1 = 6.37
    v2 = 1.43

    # Criação do objeto smartwedge_old:
    sm = smartwedge(c, r0, wc, v1, v2, Tprime, offset)

    # Obtém posição fantasma dos focos no referencial do transdutor:
    r_focus = r0 - 15
    ang_focus = 46
    focus_dist, focus_ang_deg = find_focus_coord(r_focus, ang_focus, sm)
    focal_law = contact_point_focus_focal_law(focus_dist, focus_ang_deg, sm.wedge_cl*1e3, elem_coord)
    plt.bar(elem_coord_x, focal_law, width=0.5e-3, edgecolor=[1, 0, 0])

    betas = np.arange(-60, 60, 5)
    x = betas
    y = list()
    for beta in betas:
        delay = compensate_time(beta, sm)
        y.append(delay/1e-6)

    plt.plot(x,y)
