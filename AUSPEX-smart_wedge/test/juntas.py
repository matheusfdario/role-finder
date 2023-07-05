import numpy as np
import matplotlib.pyplot as plt


def cosdeg(deg):
    return np.cos(np.pi*deg/180)


def sindeg(deg):
    return np.sin(np.pi*deg/180)


n_iter = 10

link_lengths = [10, 5, 2]
joints_pos = np.zeros((3, 2))

center = np.array([link_lengths[0], 0])
radius = 2

surf_angles = np.arange(0, -180, -5) + 180

final_x = center[0] + radius*cosdeg(surf_angles)
final_y = center[0] + radius*sindeg(surf_angles)

pos_list = list()

for i in range(len(surf_angles)):
    positions = np.zeros((4, 2))
    positions[3, 0] = final_x[i]
    positions[3, 1] = final_y[i]
    positions[2, 0] = final_x[i] + link_lengths[2] * cosdeg(surf_angles[i])
    positions[2, 1] = final_y[i] + link_lengths[2] * sindeg(surf_angles[i])

    # Notation from https://mathworld.wolfram.com/Circle-CircleIntersection.html
    R = link_lengths[1]
    r = link_lengths[2]
    d = np.sqrt((positions[3, 0] - positions[2, 0])**2 + (positions[3, 1] - positions[2, 1])**2)
    x = (d**2 - r**2 + R**2) / (2 * d)
    versor = positions[2, :] / np.linalg.norm(positions[2, :])
    positions[1, 0] = x * versor[0]
    positions[1, 1] = x * versor[1]
    positions[0, :] = [0, 0]

    plt.clf()
    plt.plot(positions[:, 0], positions[:, 1], '-o')
    plt.axis([-20, 20, -20, 20])
    plt.pause(0.05)

plt.show()
