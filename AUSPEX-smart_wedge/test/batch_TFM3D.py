import matplotlib.pyplot as plt
import numpy as np
from imaging import tfm3d
from framework.data_types import ImagingROI, ElementGeometry
from framework import file_civa, post_proc
import os
import xml.etree.ElementTree
import datetime
from surface.surface import Surface, SurfaceType
from scipy.signal import hilbert


def batch(root_path, scattering_angle=None, elem_geometry=ElementGeometry.RECTANGULAR):
    root_path = root_path.replace(os.sep, '/')
    if root_path[-4:] == '.var':
        root_path = root_path + '/proc0/results'
        is_civa_var = True
    else:
        is_civa_var = False
    listdir = os.listdir(root_path)
    surf = None
    #if True:
    for dir_path in listdir:
        #dir_path = listdir[-1]
        complete_path = os.path.join(root_path, dir_path)
        if is_civa_var:
            complete_path = complete_path + '/model.civa'
        complete_path = complete_path.replace(os.sep, '/')
        print('PATH: ' + complete_path)

        if os.path.isdir(complete_path):
            data = file_civa.read(complete_path)

            # Encontra, no arquivo model.xml, as coordenadas do centro da esfera
            tree = xml.etree.ElementTree.parse(complete_path + "/proc0/model.xml")
            root = tree.getroot()
            node = root.find("ListeDefauts/Defaut/Positioning")
            center_z = float(node[1][0].items()[0][1]) + 10.
            center_y = float(node[1][0].items()[1][1]) - 30.
            center_x = float(node[1][0].items()[2][1]) - 30.

            # Cria a ROI centrada no centro da esfera
            height = 20.0
            width = 30.0
            depth = 30.0
            h_len = 41
            w_len = 61
            d_len = 61
            datetime_str = datetime.datetime.now().__str__()[:-7].replace(':', '-')
            corner_roi = np.array([center_x-width/2, center_y-depth/2, center_z-height/2])[np.newaxis, :]
            roi = ImagingROI(corner_roi, height=height, width=width, depth=depth, h_len=h_len, w_len=w_len, d_len=d_len)

            # Executa o TFM3D
            print('Performing 3D TFM...')
            if surf is None:
                surf = Surface(data)
                surf.surfacetype = SurfaceType.LINE_NEWTON
                surf.linenewtonparam.a = 1e-10
                surf.linenewtonparam.b = 10.

            data.ascan_data = hilbert(data.ascan_data[:, :, :, 0], axis=0)[:, :, :, np.newaxis].astype(np.complex64)
            chave = tfm3d.tfm3d_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl,
                                       elem_geometry=elem_geometry, scattering_angle=scattering_angle, surf=surf)

            # Exibe e salva a figura do slice central do volume reconstruido
            fig = plt.figure()
            plt.imshow(post_proc.envelope(np.abs(data.imaging_results[chave].image[:, np.floor_divide(d_len, 2), :]), -2),
                       aspect='equal',
                       extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
            plt.show()
            plt.xlabel('x[mm]')
            plt.ylabel('z[mm]')
            title = datetime_str + '_TFM3D_elem' + elem_geometry.name + '_angle' + scattering_angle.__str__()
            plt.title(title)
            plt.savefig(complete_path + '/' + title + '.png')
            plt.close(fig)

            np.save(complete_path + '/' + title,
                    np.array([roi, data.imaging_results[chave].image]))


root_path = 'J:\Simulacoes novembro\RetangularRetangular.var'
scattering_angle = None
elem_geometry = ElementGeometry.RECTANGULAR
batch(root_path=root_path, scattering_angle=scattering_angle, elem_geometry=elem_geometry)

print('FINISHED')
