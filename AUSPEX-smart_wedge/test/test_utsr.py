import numpy as np
import ttictoc
from matplotlib import pyplot as plt
from sys import platform

from framework import file_civa, file_mat
from framework.data_types import ImagingROI
from framework.post_proc import envelope, normalize
import framework.schmerr_model as sm
from imaging import bscan, saft, wk_saft, utsr

using_civa = False
full_roi = True
result_color = False

if using_civa:
    if platform == "win32":
        # Caminhos para arquivos com simulações no Windows.
        data_point = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
    else:
        # Caminhos para arquivos com simulações no Linux.
        data_point = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")

    cl = 5900.0
    du = data_point.probe_params.pitch * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    with_noise = False
    simul_data = False
else:
    # Carrega os dados do ensaio no formato MATLAB.
    data_point = file_mat.read("../data/DadosEnsaio.mat")
    data_point.probe_params.central_freq = 4.6
    data_point.probe_params.bw = 4.4 / 4.6
    cl = 5859.4
    du = (data_point.inspection_params.step_points[1, 0] - data_point.inspection_params.step_points[0, 0]) * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    with_noise = True
    simul_data = False

# Define os parâmetros do ensaio
t1 = 100.e-9
ermv = cl / 2.0
t = ttictoc.TicToc()

# Define a ROI.
if full_roi:
    corner_roi = np.array([0.0, 0.0, 0.0])[np.newaxis, :]
    full_width = data_point.inspection_params.step_points[-1, 0] - data_point.inspection_params.step_points[0, 0]
    full_width += du / 1e-3
    roi = ImagingROI(corner_roi,
                     height=(data_point.time_grid[-1, 0] + dt / 1e-6) * cl / 2000.0,
                     h_len=data_point.inspection_params.gate_samples,
                     width=full_width,
                     w_len=data_point.inspection_params.step_points.shape[0])
else:
    if using_civa:
        corner_roi = np.array([-10.0, 0.0, 30.0])[np.newaxis, :]
    else:
        corner_roi = np.array([10.0, 0.0, 30.0])[np.newaxis, :]

    roi = ImagingROI(corner_roi, height=20.0, width=20.0, h_len=200, w_len=200)

if simul_data:
    # Parâmetros da descontinuidade para simulação.
    pos_center = 15.0
    depth_center = 40.0
    snr = 20

    # Descontinuidade (ponto infinitesimal na posição ``point_test``).
    point_test_x_idx = np.searchsorted(roi.w_points, pos_center)
    point_test_z_idx = np.searchsorted(roi.h_points, depth_center)
    point_test = np.array([roi.w_points[point_test_x_idx], 0.0, roi.h_points[point_test_z_idx]])[np.newaxis, :]
    fig_point = np.zeros((roi.h_len, roi.w_len))
    fig_point[point_test_z_idx, point_test_x_idx] = 1

    # Cria a matriz de modelo (e também o filtro casado) para um ponto infinitesimal.
    model_point, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1)
    model_sdh = None

    # Calcula índices da ROI dentro da grade de aquisição.
    # Esses valores são necessários para localizar os dados fornecidos por ``modelo_s2`` no *array* de aquisição do
    # DataInsp.
    z0 = (roi.h_points[0] - data_point.inspection_params.step_points[0, 2]) * 1e-3
    ze = (roi.h_points[-1] - data_point.inspection_params.step_points[0, 2]) * 1e-3
    idx_t0 = np.searchsorted(data_point.time_grid[:, 0], (z0 / ermv) / 1e-6)
    idx_te = np.searchsorted(data_point.time_grid[:, 0], (ze / ermv) / 1e-6)

    # Gera os sinais de A-scan a partir do modelo.
    s_t_u_point, gain = utsr.model_s2_direct(fig_point,
                                             nt0=data_point.inspection_params.gate_samples,
                                             nu0=data_point.inspection_params.step_points.shape[0],
                                             dt=(data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6,
                                             du=du,
                                             roi=roi,
                                             tau0=data_point.time_grid[idx_t0, 0],
                                             c=cl,
                                             model_transd=model_point,
                                             coord_orig=data_point.inspection_params.step_points[0])

    # Cria o sinal de ruído baseado no sinal de calibração.
    var_data_point = np.var(s_t_u_point.flatten("F"))
    sigma_n = np.sqrt(var_data_point / (10.0 ** (snr / 10.0)))

    # Coloca os valores simulados na estrutura de dados.
    if with_noise:
        ruido_ascan = sigma_n * np.random.randn(*s_t_u_point.shape)
        data_point.ascan_data = np.zeros(data_point.ascan_data.shape)
    else:
        ruido_ascan = 0.0

    data_point.ascan_data[idx_t0: idx_te + 1, 0, 0, :] = s_t_u_point + ruido_ascan

    # Calcula o parâmetro de regularização.
    alpha = np.sqrt(np.var(ruido_ascan.flatten("F")) / np.var(fig_point.flatten("F")))
else:
    # Cria a matriz de modelo (e também o filtro casado) para um SDH de 1 mm de diâmetro.
    model_sdh, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1,
                                                     flaw_type='sdh', dimmension=1e-3 / 2)
    model_point, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1)

    # Atribui um valor para o parâmetro de regularização.
    alpha = 30.0e-3

# Reconstruções da simulação.
# Faz reconstrução pelo método B-scan.
t.tic()
chave_bscan = bscan.bscan_kernel(data_point, roi=roi, c=cl)
t.toc()
print("B-scan executado em %f segundos" % t.elapsed)
image_out_bscan = normalize(envelope(data_point.imaging_results[chave_bscan].image, -2))

# Faz reconstrução pelo método SAFT.
t.tic()
chave_saft = saft.saft_kernel(data_point, roi=roi, c=cl)
t.toc()
print("SAFT executado em %f segundos" % t.elapsed)
image_out_saft = normalize(envelope(data_point.imaging_results[chave_saft].image, -2))

# Faz reconstrução pelo método wk-SAFT.
t.tic()
chave_wk_saft = wk_saft.wk_saft_kernel(data_point, roi=roi, c=cl)
t.toc()
print("wk-SAFT executado em %f segundos" % t.elapsed)
image_out_wk_saft = normalize(envelope(data_point.imaging_results[chave_wk_saft].image, -2))

# Faz reconstrução pelo método UTSR.
t.tic()
chave_utsr = utsr.utsr_kernel(data_point, roi=roi, c=cl, debug=True, alpha=alpha)
t.toc()
print("UTSR executado em %f segundos" % t.elapsed)
image_out_utsr = normalize(envelope(data_point.imaging_results[chave_utsr].image, -2))

# Plota as imagens.
if result_color:
    cmap_string = 'jet'
else:
    cmap_string = 'Greys'

fg1 = plt.figure()
im_bscan = plt.imshow(image_out_bscan, aspect='auto', cmap=plt.get_cmap(cmap_string),
                      extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("B-SCAN", {'fontsize': 8})
plt.grid(b=True, which='major', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle='--')
plt.show()

fg2 = plt.figure()
im_saft = plt.imshow(image_out_saft, aspect='auto', cmap=plt.get_cmap(cmap_string),
                     extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("SAFT", {'fontsize': 8})
plt.grid(b=True, which='major', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle='--')
plt.show()

fg3 = plt.figure()
im_wk_saft = plt.imshow(image_out_wk_saft, aspect='auto', cmap=plt.get_cmap(cmap_string),
                        extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("wk-SAFT", {'fontsize': 8})
plt.grid(b=True, which='major', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle='--')
plt.show()

fg4 = plt.figure()
im_utsr = plt.imshow(image_out_utsr, aspect='auto', cmap=plt.get_cmap(cmap_string),
                     extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("UTSR", {'fontsize': 8})
plt.grid(b=True, which='major', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle='--')
plt.show()
