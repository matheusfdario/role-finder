import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from class_smartwedge import smartwedge

# Ellipse parameters:
c = 84.28
r0 = 67.15
wc = 6.2
offset = 2
Tprime = 19.5

# Velocidade do som nos materiais em km/s
v1 = 6.37
v2 = 1.43

sm = smartwedge(c, r0, wc, v1, v2, Tprime, offset)

betas = np.arange(-60, 60+2, 2)
for beta in betas:
    alpha, intersection = sm.compute_entrypoints(beta)
    plt.scatter(*intersection, color=[1, 0, 0], s=1.5)



ax = plt.gca()
x_offset = sm.a - sm.c

specimen_circle = plt.Circle((0, 0), r0, facecolor='None', edgecolor='r')

# Sapata:
x_center = np.arange(sm.N[0], 85.24284999101062, 1e-3)
plt.plot(x_center, sm.parametric_curve(x_center), color=[0, 1, 0])
plt.plot(x_center, -sm.parametric_curve(x_center), color=[0, 1, 0])
cylinder_circle = plt.Circle((0, 0), np.linalg.norm(sm.N), facecolor='None', edgecolor=[0, 1, 0])
cylinder_circle_former = plt.Circle((0, 0), np.linalg.norm(sm.r0 + sm.wc), facecolor='None', edgecolor=[.85, .85, .85])
ellipse_externo = Ellipse((sm.a - x_offset, 0), 2 * sm.a, 2 * sm.b, facecolor='None', edgecolor=[0, 1, 0])


ax.add_patch(cylinder_circle)
ax.add_patch(ellipse_externo)
ax.add_patch(cylinder_circle_former)
ax.add_patch(specimen_circle)
plt.axis('equal')
plt.legend(loc='lower right')