import numpy as np
from matplotlib import pyplot as plt
from sys import platform

from framework import file_civa
from framework.data_types import ImagingROI
from imaging import utsr

# Faz a leitura dos dados da inspeção
if platform == "win32":
    data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
else:
    data = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")

c = 5900.0
ermv = c / 2.0

# Define a ROI.
corner_roi = np.array([-10.0, 0.0, 30.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=20.0, width=20.0, h_len=200, w_len=200)

# Define o ponto de teste.
point_test_x_idx = np.random.randint(roi.w_len)
point_test_z_idx = np.random.randint(roi.h_len)
point_test = np.array([roi.w_points[point_test_x_idx], 0.0, roi.h_points[point_test_z_idx]])[np.newaxis, :]

x0 = (roi.w_points[0] - data.probe_params.elem_center[0, 0]) * 1e-3
z0 = (roi.h_points[0] - data.probe_params.elem_center[0, 2]) * 1e-3
ze = (roi.h_points[-1] - data.probe_params.elem_center[0, 2]) * 1e-3
idx_t0 = np.searchsorted(data.time_grid[:, 0], (z0 / ermv) / 1e-6)
idx_te = np.searchsorted(data.time_grid[:, 0], (ze / ermv) / 1e-6)
with_roi = True
random_data = False

if with_roi:
    # Gera uma conjunto de dados com as dimensões do tempo compatíveis com a imagem.
    if random_data:
        # Valores aleatórios dos dados e da imagem.
        g = np.random.randn(idx_te - idx_t0 + 1, data.probe_params.num_elem)
        f = np.random.randn(roi.h_len, roi.w_len)
    else:
        # Valores dos dados retirados da estrutura DataInsp e imagem um ponto no centro da ROI.
        g = np.zeros((idx_te - idx_t0 + 1, data.probe_params.num_elem))
        for i in range(data.probe_params.num_elem):
            g[:, i] = data.ascan_data[idx_t0: idx_te + 1, i, i, 0]
        f = np.zeros((roi.h_len, roi.w_len))
        f[np.searchsorted(roi.h_points, point_test[0, 2]),
          np.searchsorted(roi.w_points, point_test[0, 0])] = 1
else:
    # Gera uma conjunto de dados e uma imagem do mesmo tamanho dos A-scan.
    if random_data:
        # Valores aleatórios dos dados e da imagem.
        g = np.random.randn(data.inspection_params.gate_samples, data.probe_params.num_elem)
        f = np.random.randn(data.inspection_params.gate_samples, data.probe_params.num_elem)
    else:
        # Valores dos dados retirados da estrutura DataInsp e imagem um ponto no centro da ROI.
        g = np.zeros((data.inspection_params.gate_samples, data.probe_params.num_elem))
        for i in range(data.probe_params.num_elem):
            g[:, i] = data.ascan_data[:, i, i, 0]
        f = np.zeros((data.inspection_params.gate_samples, data.probe_params.num_elem))
        f[np.random.randint(f.shape[0]), np.random.randint(f.shape[1])] = 1

# Aplica os operadores
if with_roi:
    f_proj, _ = utsr.model_s2_adjoint(g, c=c,
                                      nt0=data.inspection_params.gate_samples,
                                      nu0=data.probe_params.num_elem,
                                      dt=data.inspection_params.sample_time * 1e-6,
                                      du=data.probe_params.pitch * 1e-3,
                                      roi=roi, tau0=data.time_grid[idx_t0, 0],
                                      coord_orig=data.probe_params.elem_center[0, :])

    g_proj, _ = utsr.model_s2_direct(f, c=c,
                                     nt0=data.inspection_params.gate_samples,
                                     nu0=data.probe_params.num_elem,
                                     dt=data.inspection_params.sample_time * 1e-6,
                                     du=data.probe_params.pitch * 1e-3,
                                     roi=roi, tau0=data.time_grid[idx_t0, 0],
                                     coord_orig=data.probe_params.elem_center[0, :])

    ff_proj, _ = utsr.model_s2_adjoint(g_proj, c=c,
                                       nt0=data.inspection_params.gate_samples,
                                       nu0=data.probe_params.num_elem,
                                       dt=data.inspection_params.sample_time * 1e-6,
                                       du=data.probe_params.pitch * 1e-3,
                                       roi=roi, tau0=data.time_grid[idx_t0, 0],
                                       coord_orig=data.probe_params.elem_center[0, :])
else:
    f_proj, _ = utsr.model_s2_adjoint(g, c=c,
                                      nt0=data.inspection_params.gate_samples,
                                      nu0=data.probe_params.num_elem,
                                      dt=data.inspection_params.sample_time * 1e-6,
                                      du=data.probe_params.pitch * 1e-3)

    g_proj, _ = utsr.model_s2_direct(f, c=c,
                                     nt0=data.inspection_params.gate_samples,
                                     nu0=data.probe_params.num_elem,
                                     dt=data.inspection_params.sample_time * 1e-6,
                                     du=data.probe_params.pitch * 1e-3)

    ff_proj, _ = utsr.model_s2_adjoint(g_proj, c=c,
                                       nt0=data.inspection_params.gate_samples,
                                       nu0=data.probe_params.num_elem,
                                       dt=data.inspection_params.sample_time * 1e-6,
                                       du=data.probe_params.pitch * 1e-3)

# Calcula os produtos internos
dot_g = np.vdot(g_proj.flatten("F"), g.flatten("F"))
dot_f = np.vdot(f_proj.flatten("F"), f.flatten("F"))

print(dot_g, dot_f, dot_g - dot_f)

plt.figure()
plt.imshow(f_proj, aspect='auto')
plt.show()
plt.figure()
plt.imshow(f, aspect='auto')
plt.show()
plt.figure()
plt.imshow(ff_proj, aspect='auto')
plt.show()
plt.figure()
plt.imshow(g_proj, aspect='auto')
plt.show()
