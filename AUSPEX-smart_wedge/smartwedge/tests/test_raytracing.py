import numpy as np

from smartwedge.smartwedge import Smartwedge
from smartwedge.raytracing import *

if __name__ == "__main__":
    # Ellipse parameters:
    c = 84.28
    r0 = 67.15
    wc = 6.2
    offset = 2
    Tprime = 19.5

    # Velocidade do som nos materiais em mm/us
    v1 = 6.46
    v2 = 1.43
    v_steel = 5.9

    # Espessura da parede da tubulação em milímetros:
    tube_wall_width = 16

    #
    betas = np.linspace(-40, 40, 161)

    # Criação do objeto smartwedge_old:
    sm = Smartwedge(c, r0, wc, v1, v2, Tprime, offset)

    # Posição da meia-cana:
    centroMeiaCana = np.array([sm.B[1], sm.B[0]])
    raioMeiaCana = sm.r0
    epsilon = 1
    x_meiacana = np.linspace(-raioMeiaCana, raioMeiaCana - epsilon, 1500)
    z_meiacana = lambda x: (-np.sqrt(raioMeiaCana ** 2 - x ** 2) + centroMeiaCana[1])

    Sx2 = x_meiacana
    Sz2 = z_meiacana(x_meiacana)

    # Definindo a superfície 1:
    Sx1_original = np.zeros(len(betas))
    Sz1_original = np.zeros(len(betas))
    for i, beta in enumerate(betas):
        _, coord = sm.compute_entrypoints(beta)
        Sz1_original[i], Sx1_original[i] = coord
        Sz1_original[i] = sm.B[0] - Sz1_original[i]
        Sx1_original[i] = Sx1_original[i]
        plt.plot(Sx1_original, Sz1_original, 'o')

    Sx1 = np.linspace(np.min(Sx1_original), np.max(Sx1_original), 2000)
    Sz1 = np.interp(Sx1, Sx1_original, Sz1_original)

    pitch = 0.6
    # Pontos do transdutor:
    Tx = np.arange(-32 * pitch, 32 * pitch, pitch)
    Tz = np.zeros_like(Tx)

    # Pontos da ROI:
    Fx = np.array([22, 0])
    Fz = np.array([131, 168])

    # Velocidades:
    c1 = v1
    c2 = v2
    c3 = v_steel

    tolerancia = 1

    ang_critico1 = computeCriticAng(c1, c2)
    ang_critico2 = computeCriticAng(c2, c3)

    # Compute normal angle for first surf (smartwedge):
    normal1 = computeNormalAng(Sx1, Sz1)
    normal2 = computeNormalAng(Sx2, Sz2)
    
    results = cdist_arb_kernel_3medium(Fx, Fz, Sx1, Sz1, Sx2, Sz2, Tx, Tz, ang_critico1, ang_critico2, c1, c2, c3,
                                       normal1, normal2, tolerancia, plot_ray=True)
