import matplotlib.pyplot as plt
import numpy as np
from framework import file_civa
from framework.data_types import ImagingROI, ElementGeometry
from surface.surface import Surface, SurfaceType, Lineparam
from imaging import tfm
from framework import post_proc
import os
import time
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

t0 = time.time()
# path = 'C:/data/mirror/2021-03-11/Esferas/RetangularRetangular1.5mmCentro.var/proc0/results/Config_[0]/model.civa'
# data = file_civa.read(path)
# corner_roi = np.array([-6.0, -6, 20.0])[np.newaxis, :]
# roi = ImagingROI(corner_roi, height=10.0, width=12.0, depth=12.0, h_len=60, w_len=60, d_len=61)
# print('Creating Surface...')
# surf = Surface(data, -1, surf_type=SurfaceType.LINE_NEWTON, surf_param=Lineparam(a=0.0, b=10.0, SSE=1e-9))
# surf.fit(surf_type=SurfaceType.LINE_NEWTON, surf_param=Lineparam(a=1e-6, b=10.0, SSE=1e-9))
# scattering_angle = 12
# print('Performing TFM...')
# chave = tfm.tfm3d_kern(data, roi=roi, sel_shot=0, c=data.specimen_params.cl,
#                        elem_geometry=ElementGeometry.RECTANGULAR, surf=surf,
#                        scattering_angle=scattering_angle)
# print('Copying result...')
# tfm = np.copy(data.imaging_results[chave].image)
# print('Saving result...')
# saved = np.array([tfm, roi], dtype=object)
# np.save(path+'/'+str(time.time()), saved)
# print('Processing result...')
tfm = np.load('C:/data/mirror/2021-03-11/Esferas/RetangularRetangular2.5mmCentro.var/2021-03-11-09-23-17/TFM pitch=3.2mm scatangle=None2021-03-11-09-26-12.npy', allow_pickle=True)
envel = np.abs(tfm[0])
roi = tfm[1]
cscan = envel.argmax(0)
cscan = envel.shape[0] - cscan
cscan = cscan * (envel.max(0) > envel.max()/10)
#inv = roi.h_len - cscan

fig = plt.figure()
ax = fig.gca(projection='3d')

X = roi.w_points
Y = roi.d_points
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = cscan

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


print('Elapsed time [s] ' + (time.time() - t0).__str__())