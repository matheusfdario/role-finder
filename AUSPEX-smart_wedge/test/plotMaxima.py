import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
import os
from datetime import datetime


def draw_sphere(r, c, ax):
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:50j, 0:np.pi:10j]
    x = c[0] + r * np.cos(u) * np.sin(v)
    y = c[1] + r * np.sin(u) * np.sin(v)
    z = c[2] + r * np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", linewidth=.5, alpha=.5)


def draw_spheres(ax):
    r = 2.5
    z = 25 + 3
    #z = 25 + r

    # draw_sphere(r=10, c=[5, 5, 35], ax=ax)

    # draw_sphere(r=r, c=[5, 5, z], ax=ax)
    # draw_sphere(r=r, c=[5, -5, z], ax=ax)
    # draw_sphere(r=r, c=[-5, 5, z], ax=ax)
    # draw_sphere(r=r, c=[-5, -5, z], ax=ax)
    # draw_sphere(r=r, c=[0, 0, z], ax=ax)

    # draw_sphere(r=r, c=[-5, 0, z], ax=ax)
    # draw_sphere(r=r, c=[0, -5, z], ax=ax)
    # draw_sphere(r=r, c=[0, 5, z], ax=ax)
    # draw_sphere(r=r, c=[5, 0, z], ax=ax)

def the_batch(pathroot, timestamp, sub=None, figlist=None, threshold=.5, amplitude=False):
    xdczerototal = -1
    now = now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '-')
    path = pathroot + '/' + timestamp
    # Counts the number of .npy files as loads the filenames
    # to a list (called, guess what, filenames)
    filenames = list()
    dir_res = os.listdir(path)
    for elem in dir_res:
        if elem[-4:] == '.npy' and elem[:3] == 'TFM':
            filenames.append(elem)
    N = len(filenames)
    max_pixel = np.zeros(N)
    ratios = np.zeros(N)

    pitch_list = np.load(path + '/pitch_list.npy')

    if figlist is None:
        figlist = list()
    #plt.figure(figsize=(8, 6), dpi=200)
    for i_config in range(N):
        saved = np.load(path + '/' + filenames[i_config], allow_pickle=True)
        tfm = saved[0]
        roi = saved[1]
        envel = np.abs(tfm)
        cscan = roi.h_points[envel.argmax(0)]
        #cscan = roi.h_points[-1] - cscan
        #minimum = cscan[(envel.max(0) > envel.max() * threshold)].max()
        minimum = 25. + 12.
        cscan = cscan * (envel.max(0) > envel.max() * threshold) + \
                (envel.max(0) <= envel.max() * threshold) * minimum
        # inv = roi.h_len - cscan


        side = str(np.round(pitch_list[i_config] - .1, 2))
        if sub is None:
            fig = plt.figure(figsize=(8, 6), dpi=200)
            figlist.append([fig])
            ax = fig.gca(projection='3d')
            #ax.set_title('side: ' + side + ' mm')
        elif sub == 1:
            fig = plt.figure(figsize=(8, 6), dpi=200)
            figlist.append([fig])
            ax = fig.add_subplot(121, projection='3d')
            #ax.set_title('side: ' + side + ' mm, maxangle: 180')
        else:
            fig = figlist[i_config][0]
            ax = fig.add_subplot(122, projection='3d')
            #ax.set_title('side: ' + side + ' mm, maxangle: 12')

        X = roi.w_points
        Y = roi.d_points
        X, Y = np.meshgrid(X, Y)
        Z = cscan
        maxima = envel.max(0)
        if amplitude is True:
            my_col = cm.coolwarm_r(maxima)
            surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=my_col,
                                   linewidth=0, antialiased=True)
            fig.colorbar(
                cm.ScalarMappable(norm=colors.Normalize(vmin=maxima.min(), vmax=maxima.max()),
                                  cmap='coolwarm_r'), ax=ax)
        else:
            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm_r,
                                   linewidth=0, antialiased=True)

        limits = [ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]

        draw_spheres(ax)

        ax.set_xlim3d(limits[0])
        ax.set_ylim3d(limits[1])
        #ax.set_zlim3d(limits[2])
        ax.set_zlim3d((25.0, 30.0))
        ax.elev = -140.0
        ax.azim = 45.
        ax.grid(False)
        max_filtered = maxima * (envel.max(0) > envel.max() * threshold)
        ax.set_title('th='+str(threshold) + ' ' + 'max=' + '{:.3f}'.format(max_filtered.max()))

        # Draw a projection of the transducer
        # l = (pitch_list[i_config] * 11 - .1) / 2
        # x = [-l, -l, l, l, -l]
        # y = [-l, l, l, -l, -l]
        # z = [25] * 5
        # ax.plot(x, y, z)

        plt.show()
        if sub == 2:
            figpath = path + '/' + filenames[i_config] + ' ' + now + ' ' + str(i_config) +'.png'
            print(figpath)
            fig.savefig(figpath)

    return figlist

#figlist = the_batch('C:/data/mirror/2021-01-26/RetangularRetangular2.5mmCentro.var', '2021-01-27-00-23-18', 1)
#the_batch('C:/data/mirror/2021-01-26/RetangularRetangular2.5mmCentro.var', '2021-01-27-00-42-44', 2, figlist)

#figlist = the_batch('D:/2021-02-11/RetangularRetangular2.5mmCentro22.var', '2021-02-11-18-53-46', 1)
#the_batch('D:/2021-02-11/RetangularRetangular2.5mmCentro22.var', '2021-02-11-18-58-06', 2, figlist)

#figlist = the_batch('D:/2021-02-11/RetangularRetangular2.5mmCentro6x2.var', '2021-02-11-20-32-36', 1)
#the_batch('D:/2021-02-11/RetangularRetangular2.5mmCentro6x2.var', '2021-02-11-20-37-57', 2, figlist)

#the_batch('C:/data/mirror/2021-03-11/Esferas/RetangularRetangular2.5mmCentro.var', '2021-03-11-09-23-17')

for th in [1/2, 1/4, 1/8]:
    figlist = the_batch('C:/data/mirror/2021-03-18/RetangularRetangular2.5mmRet.var', '2021-03-19-08-10-10', sub=1, threshold=th, amplitude=False)
    the_batch('C:/data/mirror/2021-03-18/RetangularRetangular2.5mmRet.var', '2021-03-19-08-13-22', sub=2, figlist=figlist, threshold=th, amplitude=False)