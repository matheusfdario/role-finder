import numpy as np


from framework import file_civa
from framework.data_types import ImagingROI, ImagingResult
from imaging import saft

# Cria a ROI
corner_roi = np.array([-10.0, 0.0, 30.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=20.0, width=20.0, h_len=200, w_len=200)

# Carrega o vetor x (uma imagem aleatória)
x = ImagingResult(roi)
x.image = np.random.randn(roi.h_len, roi.w_len)

# Carrega o vetor y (dados aleatórios com o mesmo tamanho dos dados de A-scan do ultrassom)
data = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
y = np.random.randn(data.inspection_params.gate_samples, data.probe_params.num_elem)

# Calcula o vetor ÿ = Ax (operador direto).
ytil = saft.saft_oper_direct(x.image, roi, data.probe_params.elem_center, data.ascan_data.shape[0],
                              data.probe_params.elem_center.shape[0])

# Calcula o vetor ẍ = A'y (operador adjunto).
xtil = saft.saft_oper_adjoint(y, roi, data.probe_params.elem_center)

# Faz o teste do produto interno.
dot_product_y_ytil = np.dot(y.flatten('F'), ytil.flatten('F'))
dot_product_xtil_x = np.dot(xtil.flatten('F'), x.image.flatten('F'))
print(dot_product_y_ytil, " - ", dot_product_xtil_x, " = ", dot_product_y_ytil - dot_product_xtil_x)
