import matplotlib.pyplot as plt
import numpy as np
import time
from surface.surface import Surface, SurfaceType, Lineparam
from framework import file_m2k, file_mat, file_civa, file_omniscan
from scipy.signal import hilbert
from imaging import tfm  # adicionado tfm3d
from framework.data_types import ImagingROI, ElementGeometry  # ImagingROI3D não existe
from framework import file_m2k, file_civa, post_proc
import os
from datetime import datetime


def the_bacth(pathroot, doi=0.0, scattering_angle=None):
    xdczerototal = -1

    now = str(datetime.now())[:-7].replace(':', '-').replace(' ', '-')
    #pathroot = 'D:/RetangularRetangularLadoVel.var'
    os.mkdir(pathroot + '/' + now)

    print('Creating ROI...')

    # For point spread function
    #corner_roi = np.array([-15.0, -15, 15.0])[np.newaxis, :]
    #roi = ImagingROI(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=61)

    # For spheres
    #corner_roi = np.array([-6.0, -6, 18.0])[np.newaxis, :]
    #roi = ImagingROI(corner_roi, height=12.0, width=12.0, depth=12.0, h_len=60, w_len=60, d_len=61)

    # For further spheres
    # corner_roi = np.array([7.0, -5, 23.0])[np.newaxis, :]
    # roi = ImagingROI(corner_roi, height=10.0, width=23.0, depth=10.0, h_len=100/2, w_len=230/2, d_len=100/2)

    corner_roi = np.array([18.0, -5, 21.0])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=10.0, width=10.0, depth=10.0, h_len=100 / 2, w_len=100 / 2, d_len=100 / 2)

    # Nearest neighbor
    i_depth = np.argmin(np.abs(roi.d_points - doi))

    # Counts the number of subdirectories in the CIVA var folder
    N = 0
    dir_res = os.listdir(pathroot + '/proc0/results/')
    for elem in dir_res:
        if elem[-1] == ']':
            N += 1

    pitch_list = np.zeros(N)

    for i_config in np.arange(N):
        t0 = time.time()

        print('Loading simulation data...')
        path = pathroot + '/proc0/results/Config_[' + i_config.__str__() + ']/model.civa'
        data = file_civa.read(path)
        pitch = np.round(np.linalg.norm(data.probe_params.elem_center[0, :]
                                        - data.probe_params.elem_center[1, :]), 1)
        pitch_list[i_config] = pitch

        print('Performing 3D TFM...')
        b = data.inspection_params.water_path
        if 'surf' not in locals():
            surf = Surface(data, xdczerototal, surf_type=SurfaceType.LINE_NEWTON, surf_param=Lineparam(a=0.0, b=b, SSE=1e-9))
            surf.fit(surf_type=SurfaceType.LINE_NEWTON, surf_param=Lineparam(a=1e-6, b=b, SSE=1e-9))
        chave = tfm.tfm3d_kern(data, roi=roi, sel_shot=0, c=data.specimen_params.cl,
                               elem_geometry=ElementGeometry.RECTANGULAR, surf=surf,
                               scattering_angle=scattering_angle)
        # tfm alterado para tfm3d
        fig = plt.figure()
        plt.imshow(np.abs(data.imaging_results[chave].image[:,i_depth,:]), aspect='equal',
                   extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
        print('Saving data...')

        now_ = str(datetime.now())[:-7].replace(':', '-').replace(' ', '-')
        title = 'TFM pitch=' + str(pitch) + 'mm' + ' scatangle=' + str(scattering_angle) + now_

        saved = np.array([data.imaging_results[chave].image, roi, scattering_angle], dtype=object)
        np.save(pathroot + '/' + now + '/' + title, saved)
        plt.show()
        plt.xlabel('x[mm]')
        plt.ylabel('z[mm]')
        plt.title(title)
        print('Saving figure...')
        plt.savefig(pathroot + '/' + now + '/' + title+'.png')
        plt.close(fig)

        del data

        print('Result ' + (i_config+1).__str__()+'/' + str(N) + '. Elapsed time [s] ' + (time.time() - t0).__str__())

    np.save(pathroot + '/' + now + '/pitch_list', pitch_list)

    print('FINISHED ' + pathroot)

the_bacth('C:/data/mirror/2021-03-19/RetangularRetangular2.5mmSec.var', scattering_angle=None)
the_bacth('C:/data/mirror/2021-03-19/RetangularRetangular2.5mmSec.var', scattering_angle=12)
the_bacth('C:/data/mirror/2021-03-19/RetangularRetangular2.5mmRet.var', scattering_angle=None)
the_bacth('C:/data/mirror/2021-03-19/RetangularRetangular2.5mmRet.var', scattering_angle=12)
# the_bacth('C:/data/mirror/2021-03-11 diam50/Esferas/RetangularRetangular5mmCentro.var', scattering_angle=None)
# the_bacth('C:/data/mirror/2021-03-11 diam50/Esferas/RetangularRetangular5mmCentro.var', scattering_angle=12)
# the_bacth('C:/data/mirror/2021-03-11 diam50/Esferas/RetangularRetangular1.5mmCentro.var', scattering_angle=None)
# the_bacth('C:/data/mirror/2021-03-11 diam50/Esferas/RetangularRetangular1.5mmCentro.var', scattering_angle=12)
# the_bacth('C:/data/mirror/2021-03-11 diam50/Esferas/RetangularRetangular2.5mmCentro.var', scattering_angle=None)
# the_bacth('C:/data/mirror/2021-03-11 diam50/Esferas/RetangularRetangular2.5mmCentro.var', scattering_angle=12)