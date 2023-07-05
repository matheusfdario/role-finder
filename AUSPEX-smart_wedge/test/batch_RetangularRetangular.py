import matplotlib.pyplot as plt
import numpy as np
import time
from surface.surface import Surface, SurfaceType
from framework import file_m2k, file_mat, file_civa, file_omniscan
from scipy.signal import hilbert
from imaging import tfm, tfm3d  # adicionado tfm3d
from framework.data_types import ImagingROI, ImagingROI3D, ElementGeometry  # ImagingROI3D não existe
from framework import file_m2k, file_civa, post_proc
import sys
import gc




xdczerototal = -1




for i_config in range(30):
    t0 = time.time()

    pitch = float(i_config)/10 + 0.1
    title = 'Rect Rect pitch=' + pitch.__str__() + 'mm'

    print('Loading simulation data...')
    data = file_civa.read('G:/RetangularRetangular.var/proc0/results/Config_[' + i_config.__str__() + ']/model.civa')

    # Deixa as coordenadas dos elementos no mesmo formato dos transdutores lineares
    N_elem = data.probe_params.elem_center.shape[0]*data.probe_params.elem_center.shape[1]
    extended = np.zeros((N_elem, 3))
    extended[:,:-1] = data.probe_params.elem_center.reshape((N_elem, 2))
    data.probe_params.elem_center = extended
    # Insere o número de elementos
    data.probe_params.num_elem = N_elem

    print('Creating ROI...')
    corner_roi = np.array([-15.0, -15, 15.0])[np.newaxis, :]
    roi = ImagingROI3D(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=61)

    print('Performing 3D TFM...')
    chave = tfm3d.tfm3d_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl, elem_geometry=ElementGeometry.RECTANGULAR)#, scattering_angle=20)
    # tfm alterado para tfm3d
    fig = plt.figure()
    plt.imshow(post_proc.envelope(data.imaging_results[chave].image[:,30,:], -2), aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
    print('Saving data...')
    np.save(title, data.imaging_results[chave].image)
    plt.show()
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title(title)
    print('Saving figure...')
    plt.savefig(title+'.png')
    plt.close(fig)

    del data

    print('Result ' + (i_config+1).__str__()+'/30' + '. Elapsed time [s] ' + (time.time() - t0).__str__())


print('FINISHED')