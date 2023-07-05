import numpy as np


from framework import file_civa
from imaging import utsr

# Cria a ROI
# corner_roi = np.array([-10.0, 0.0, 30.0])[np.newaxis, :]
# roi = ImagingROI(corner_roi, height=20.0, width=20.0, h_len=200, w_len=200)

# Carrega o vetor x (dados aleatórios com o mesmo tamanho dos dados de A-scan do ultrassom)
data = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
x = np.random.randn(data.inspection_params.gate_samples, data.probe_params.num_elem)

# Carrega o vetor y (uma imagem aleatória)
# y = ImagingResult(roi)
# y.image = np.random.randn(roi.h_len, roi.w_len)
y = np.random.randn(data.inspection_params.gate_samples, data.probe_params.num_elem)

# Calcula o vetor ÿ = Ax (operador direto).
ytil, _ = utsr.model_s2_direct(x, nt0=data.inspection_params.gate_samples, nu0=data.probe_params.num_elem)

# Calcula o vetor ẍ = A'y (operador adjunto).
# xtil, _ = utsr.model_s2_adjoint(y.image, roi)
xtil, _ = utsr.model_s2_adjoint(y, nt0=data.inspection_params.gate_samples, nu0=data.probe_params.num_elem)

# Faz o teste do produto interno.
# dot_product_y_ytil = np.dot(y.image.flatten('F'), ytil.flatten('F'))
dot_product_y_ytil = np.dot(y.flatten('F'), ytil.flatten('F'))
dot_product_xtil_x = np.dot(xtil.flatten('F'), x.flatten('F'))
print(dot_product_y_ytil, " - ", dot_product_xtil_x, " = ", dot_product_y_ytil - dot_product_xtil_x)
