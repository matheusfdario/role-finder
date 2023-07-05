import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from datetime import datetime
from smartwedge.smartwedge import Smartwedge
from smartwedge.generate_law import generate_law, contact_planewave_focal_law, compute_focus_dist

# Especificações do transdutor:
# Imasonic
pitch = 0.6e-3  # em mm
coord = np.zeros((64, 2))
coord[:, 0] = np.linspace(-63 * pitch/2, 63 * pitch/2, 64)

# Parâmetros da Smartwedge:
c = 84.28
r0 = 71
wc = 6.2
offset = 2
Tprime = 19.5

# Velocidade do som nos materiais em mm/us
v1 = 6.36
v2 = 1.43

# Criação do objeto smartwedge_old:
sm = Smartwedge(c, r0, wc, v1, v2, Tprime, offset)

# Ângulos da tubulação que irão ser varridos. O ângulo de 90 graus é equivalente às 12h e -90 graus às 6h:
beg_angle = -42
end_angle = 42
step = 2
betas = np.arange(beg_angle, end_angle+step, step)
planewave_angle = list()

for beta in betas:
    alpha, intersection = sm.compute_entrypoints(beta)
    plt.scatter(*intersection)
    planewave_angle.append(np.rad2deg(alpha))

# Gera as leis focais:
focal_law = contact_planewave_focal_law(planewave_angle, v1*1e3, coord[:, 0])

# Gera o .law:
date = datetime.today().strftime('%d-%m-%Y')
filename = "focal_law/planewave_-42_42_step" + str(step) + "_" + date
generate_law(filename, focal_law)

table = PrettyTable(["raio [mm]", "alpha [graus]"])
#
focus_dist = np.zeros_like(betas, dtype=float)
for i, ang in enumerate(betas):
    focus_dist[i] = compute_focus_dist(sm.r0, ang, sm)
    table.add_row([focus_dist[i]*1e3, planewave_angle[i]])
print(table)
# Verificar final onda plana se se todos os pontos tem mesma distância