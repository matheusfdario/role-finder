from surface.surface import Surface, SurfaceType
from framework import file_civa
import matplotlib.pyplot as plt
import numpy as np
from framework.data_types import ImagingROI

# --- Dados ---
# Carrega os dados de inspeção do arquivo de simulação do CIVA. A simulação
# considera um bloco de aço-carbono distante 15 mm da superfície do transdutor.
data = file_civa.read('block_sdh_immersion_close.civa')

# --- Surface ---
# Instancia um objeto Surface a partir dos dados. Na chamada
# do construtor Surface(), a superfície é identificada a partir
# dos dados e pode ser utilizada daqui em diante.
mySurf = Surface(data, 0, c_medium=1498, keep_data_insp=False)

# --- ROI ---
# Define uma ROI de 40 mm x 40 mm.
heigth = 40.0
width = 40.0
h_len = 100
w_len = 64
corner_roi = np.array([-20.0, 0.0, 0.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=heigth, width=width, h_len=h_len, w_len=w_len)

# --- Distâncias ---
# Calcula as distâncias percorridas na água e no meio para cada par elemento-pixel.
# Como as distâncias são dadas em mm, multiplica-se o resultado por 1e-3 para se
# obterem os valores em m.
[dist_water, dist_material] = mySurf.cdist_medium(
                data.probe_params.elem_center, roi.get_coord()) * 1e-3

# --- Tempos de percurso ---
# Calcula o tempo de percurso entre cada pixel e o elemento central do trandutor.
# O tempo de percurso em cada meio é dado pelas distâncias em cada meio divididas
# pelas respectivas velocidades de propagação das ondas longitudinais.
center_elem = int(data.probe_params.num_elem / 2)
travel_time_center_elem = dist_water[center_elem, :] / data.inspection_params.coupling_cl + \
              dist_material[center_elem, :] / data.specimen_params.cl


# --- Exibição dos tempos de persurso ---
# Exibe os tempos de percurso entre cada pixel e o elemento central do trandutor.
travel_time_center_elem = travel_time_center_elem.reshape(w_len, h_len)
travel_time_center_elem = travel_time_center_elem.transpose()
plt.imshow(travel_time_center_elem, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.colorbar()
plt.title('Tempos de percurso [s] (elemento central)', fontsize=14)

# --- Exibição da superfície ---
# A superfície encontrada pelo construtor da classe é exibida como uma linha vermelha.
plt.plot(mySurf.x_discr, mySurf.z_discr, 'r')
plt.axis([roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])

plt.show()