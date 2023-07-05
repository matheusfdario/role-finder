import numpy as np
from matplotlib import pyplot as plt

from framework import file_civa
from framework.data_types import ImagingROI, ImagingResult
from framework.post_proc import envelope, normalize, api
from imaging import saft, tfm, bscan

# Define a ROI.
corner_roi = np.array([-10.0, 0.0, 30.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=20.0, width=20.0, h_len=200, w_len=200)

# Faz a leitura dos dados da inspeção
data = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")

# Pega somente os dados de varredura para o SAFT.
g = np.zeros((data.inspection_params.gate_samples, data.probe_params.num_elem))
for i in range(data.probe_params.num_elem):
    g[:, i] = data.ascan_data[:, i, i, 0]

# Aplica o operador adjunto para se obter uma imagem.
img = saft.saft_oper_adjoint(g, roi, data.probe_params.elem_center)

# Aplica o operador direto para se obter os dados.
g_proj = saft.saft_oper_direct(img, roi, data.probe_params.elem_center, data.ascan_data.shape[0],
                                data.probe_params.elem_center.shape[0])

# Aplica o operador adjunto para se obter uma imagem.
img_proj = saft.saft_oper_adjoint(g_proj, roi, data.probe_params.elem_center)


plt.imshow(img, aspect='auto')
plt.show()

plt.imshow(img_proj, aspect='auto')
plt.show()
