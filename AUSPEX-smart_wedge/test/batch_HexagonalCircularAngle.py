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




for i_config in range(2,31,2):
    t0 = time.time()

    radius = i_config/20
    radius_str = radius.__str__()
    title = 'Hexag Circ pitch=' + radius_str + 'mm'

    print('Loading simulation data...')
    data = file_civa.read('G:/Hexagonal Circular/HexagonalCircularRadius_'+radius_str+'.civa')

    print('Creating ROI...')
    corner_roi = np.array([-15.0, -15, 15.0])[np.newaxis, :]
    roi = ImagingROI3D(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=61)

    print('Performing 3D TFM with scattering_angle...')
    title = title + '_Angle12'
    chave = tfm3d.tfm3d_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl,  # tfm alterado para tfm3d.
                             elem_geometry=ElementGeometry.CIRCULAR, scattering_angle=12)

    fig = plt.figure()
    plt.imshow(post_proc.envelope(data.imaging_results[chave].image[:, 30, :], -2), aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
    print('Saving data...')
    np.save(title, data.imaging_results[chave].image)
    plt.show()
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title(title)
    print('Saving figure...')
    plt.savefig(title + '.png')
    plt.close(fig)
    del data

    print('Result ' + (i_config+1).__str__()+'/30' + '. Elapsed time [s] ' + (time.time() - t0).__str__())


print('FINISHED')