import numpy as np
import ttictoc
from matplotlib import pyplot as plt
from sys import platform

from framework import file_civa, file_mat, file_m2k
from framework.data_types import ImagingROI
from framework.post_proc import envelope, normalize
import framework.schmerr_model as sm
from imaging import wk_saft, utsr, utsr_fista, sparsa

# ``data_type`` pode ser "CIVA", "M2K", "MATLAB", "SIMUL"
data_type = "SIMUL"
full_roi = False
result_color = False
debug = True
processing_utsr = False
processing_utsr_fista = False
processing_sparsa = True

if data_type == "CIVA":
    if platform == "win32":
        # Caminhos para arquivos com simulações no Windows.
        # data_point = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
        data_point = file_civa.read("C:/Users/asros/Documents/dados civa/Furo40mmPA_FMC_Contact_new.civa")
    else:
        # Caminhos para arquivos com simulações no Linux.
        data_point = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")

    cl = 5900.0
    du = data_point.probe_params.pitch * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    with_noise = False
    simul_data = False

elif data_type == "M2K":
    if platform == "win32":
        # Caminhos para arquivos com simulações no Windows.
        # data_point = file_m2k.read("C:/Users/GiovanniAlfredo/Desktop/ArquivosM2K/CP1_Bot_50_Direct.m2k",
        #                            type_insp="contact", water_path=0.0,
        #                            freq_transd=5.0, bw_transd=0.5, tp_transd='gaussian')
        data_point = file_m2k.read("C:/Users/asros/Documents/dados m2k/CP1_Bot_50_Direct.m2k",
                                   type_insp="contact", water_path=0.0,
                                   freq_transd=5.0, bw_transd=0.5, tp_transd='gaussian')

    else:
        # Caminhos para arquivos com simulações no Linux.
        data_point = file_m2k.read("/home/giovanni/Documents/ArquivosM2K/CP1_Bot_50_Direct.m2k",
                                   type_insp="contact", water_path=0.0,
                                   freq_transd=5.0, bw_transd=0.5, tp_transd='gaussian')

    cl = 6150.0
    du = data_point.probe_params.pitch * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    with_noise = False
    simul_data = False

elif data_type == "MATLAB":
    # Carrega os dados do ensaio no formato MATLAB.
    data_point = file_mat.read("../data/DadosEnsaio.mat")
    data_point.probe_params.central_freq = 4.6
    data_point.probe_params.bw = 4.4 / 4.6
    cl = 5859.4
    du = (data_point.inspection_params.step_points[1, 0] - data_point.inspection_params.step_points[0, 0]) * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    with_noise = False
    simul_data = False

else:
    # Carrega os dados do ensaio no formato MATLAB, somente para ter os mesmos parâmetros de inspeção, do espécime e do
    # transdutor. Os A-scan serão gerados por simulação.
    data_point = file_mat.read("../data/DadosEnsaio.mat")
    data_point.probe_params.central_freq = 4.6
    data_point.probe_params.bw = 4.4 / 4.6
    cl = 5900.0
    du = (data_point.inspection_params.step_points[1, 0] - data_point.inspection_params.step_points[0, 0]) * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    # Define a simulação.
    with_noise = True
    simul_data = True

# Define os parâmetros do ensaio
t1 = 100.e-9
ermv = cl / 2.0
t = ttictoc.TicToc()
if result_color:
    cmap_string = 'jet'
else:
    cmap_string = 'Greys'

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
    if data_type == "CIVA":
        corner_roi = np.array([-5.0, 0.0, 35.0])[np.newaxis, :]
        roi = ImagingROI(corner_roi, height=10.0, width=10.0, h_len=200, w_len=200)
    elif data_type == "M2K":
        corner_roi = np.array([-5.0, 0.0, 35.0])[np.newaxis, :]
        roi = ImagingROI(corner_roi, height=10.0, width=10.0, h_len=200, w_len=200)
    elif data_type == "MATLAB":
        corner_roi = np.array([10.0, 0.0, 35.0])[np.newaxis, :]
        roi = ImagingROI(corner_roi, height=10.0, width=10.0, h_len=200, w_len=200)
    else:
        corner_roi = np.array([10.0, 0.0, 35.0])[np.newaxis, :]
        roi = ImagingROI(corner_roi, height=10.0, width=10.0, h_len=200, w_len=200)

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
    idx_t0 = np.floor((z0 / ermv) / dt + 0.5).astype(int)
    idx_te = np.floor((ze / ermv) / dt + 0.5).astype(int)

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

    # Cálculo de alpha baseado no ruído
    alpha_noise = np.sqrt(np.var(ruido_ascan.flatten("F")) / np.var(fig_point.flatten("F")))

else:
    # Cria a matriz de modelo (e também o filtro casado) para um SDH de 1 mm de diâmetro.
    model_sdh, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1,
                                                     flaw_type='sdh', dimmension=1e-3 / 2)
    model_point, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1)

    # Atribui um valor para o parâmetro de regularização.
    alpha_noise = -1

# Define os parâmetros de cada algoritmo.
if data_type == "CIVA":
    alpha_utsr = 0.03
    alpha_utsr_fista = 0.00175
    alpha_sparsa = 0.00175

    tol_utsr = 5e-2
    tol_utsr_fista = 1e-2
    tol_sparsa = 1e-5

elif data_type == "M2K":
    alpha_utsr = 0.03
    alpha_utsr_fista = 2.25
    alpha_sparsa = 2.25

    tol_utsr = 5e-2
    tol_utsr_fista = 1e-2
    tol_sparsa = 1e-5

elif data_type == "MATLAB":
    alpha_utsr = 0.01
    alpha_utsr_fista = 0.01
    alpha_sparsa = 0.01

    tol_utsr = 4e-2
    tol_utsr_fista = 1e-3
    tol_sparsa = 1e-5

else:
    alpha_utsr = -1
    alpha_utsr_fista = -1
    alpha_sparsa = -1

    tol_utsr = 4e-2
    tol_utsr_fista = 1e-3
    tol_sparsa = 1e-5

# Reconstruções da simulação.
# Faz reconstrução pelo método wk-SAFT.
t.tic()
chave_wk_saft = wk_saft.wk_saft_kernel(data_point, roi=roi, c=cl)
t.toc()
print("wk-SAFT executado em %f segundos" % t.elapsed)
image_out_wk_saft = normalize(envelope(data_point.imaging_results[chave_wk_saft].image, -2))

# Plota a imagem.
fg3 = plt.figure()
im_wk_saft = plt.imshow(image_out_wk_saft, aspect='auto', cmap=plt.get_cmap(cmap_string),
                        extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("wk-SAFT", {'fontsize': 8})
plt.grid(b=True, which='major', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', linestyle='--')

# Faz reconstrução pelo método UTSR.
if processing_utsr:
    t.tic()
    chave_utsr = utsr.utsr_kernel(data_point, roi=roi, c=cl, debug=debug, alpha=alpha_utsr, tol=tol_utsr)
    t.toc()
    print("UTSR executado em %f segundos" % t.elapsed)
    image_out_utsr = normalize(envelope(data_point.imaging_results[chave_utsr].image, -2))

    # Plota a imagem.
    fg4 = plt.figure()
    im_utsr = plt.imshow(image_out_utsr, aspect='auto', cmap=plt.get_cmap(cmap_string),
                         extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
    plt.title("UTSR", {'fontsize': 8})
    plt.grid(b=True, which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--')

# Faz reconstrução pelo método UTSR_fista.
if processing_utsr_fista:
    t.tic()
    chave_utsr_fista = utsr_fista.utsr_fista_kernel(data_point, roi=roi, c=cl, debug=debug,
                                                    alpha=alpha_utsr_fista, tol=tol_utsr_fista)
    t.toc()
    print("UTSR-FISTA executado em %f segundos" % t.elapsed)
    image_out_utsr_fista = normalize(envelope(data_point.imaging_results[chave_utsr_fista].image, -2))

    # Plota a imagem.
    fg5 = plt.figure()
    im_utsr_fista = plt.imshow(image_out_utsr_fista, aspect='auto', cmap=plt.get_cmap(cmap_string),
                               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
    plt.title("UTSR-FISTA", {'fontsize': 8})
    plt.grid(b=True, which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--')

# Faz reconstrução pelo método SpaRSA.
if processing_sparsa:
    t.tic()
    chave_sparsa = sparsa.sparsa_kernel(data_point, roi=roi, c=cl, debug=debug, alpha=alpha_sparsa, tol=tol_sparsa)
    t.toc()
    print("SpaRSA executado em %f segundos" % t.elapsed)
    image_out_sparsa = normalize(envelope(data_point.imaging_results[chave_sparsa].image, -2))

    # Plota a imagem.
    fg6 = plt.figure()
    im_sparsa = plt.imshow(image_out_sparsa, aspect='auto', cmap=plt.get_cmap(cmap_string),
                           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
    plt.title("SpaRSA", {'fontsize': 8})
    plt.grid(b=True, which='major', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', linestyle='--')

# Mostra as imagens.
plt.show()
