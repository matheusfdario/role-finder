import numpy as np
from matplotlib import pyplot as plt
from framework import file_civa
from framework.data_types import ImagingROI
from framework.post_proc import envelope, normalize
import framework.schmerr_model as sm
from imaging import bscan, sparsa

# --- Dados ---
# Carrega os dados de inspeção do arquivo de simulação do CIVA.
data_type = "CIVA"
full_roi = False
result_color = True
debug = True
processing_sparsa = True

if data_type == "CIVA":
    # Caminhos para arquivos com simulações no Windows.
    data_point = file_civa.read("Furo40mmPA_FMC_Contact_new.civa")

    cl = 5900.0
    du = data_point.probe_params.pitch * 1e-3
    dt = (data_point.time_grid[1, 0] - data_point.time_grid[0, 0]) * 1e-6

    with_noise = False
    simul_data = False

# Define os parâmetros do ensaio
t1 = 100.e-9
ermv = cl / 2.0

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

# Cria a matriz de modelo (e também o filtro casado) para um SDH de 1 mm de diâmetro.
model_sdh, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1,
                                                 flaw_type='sdh', dimmension=1e-3 / 2)
model_point, _, _, _, _ = sm.generate_model_filter(data_point, c=cl, t1=t1)

# Atribui um valor para o parâmetro de regularização.
alpha_noise = -1

# Define os parâmetros de cada algoritmo.
if data_type == "CIVA":
    alpha_sparsa = 0.00175
    tol_sparsa = 1e-5

# --- Processamento ---
# Obtém a imagem B-scan. Note que o algoritmo retorna apenas a chave de
# identificação, sendo que o resultado é salvo na própria variável "data".
# Além disso, o algoritmo obtém a imagem na ROI definida acima.
chave_bscan = bscan.bscan_kernel(data_point, roi=roi, c=cl)

# Faz reconstrução pelo método UTSR FISTA.
if processing_sparsa:
    chave_sparsa = sparsa.sparsa_kernel(data_point, roi=roi, c=cl, debug=debug,
                                        alpha=alpha_sparsa, tol=tol_sparsa)

# --- Imagens ---
plt.figure(figsize=(16, 7))

# Exibe o resultado do algoritmo B-scan.
plt.subplot(1, 3, 1)
plt.imshow(data_point.imaging_results[chave_bscan].image, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('B-scan', fontsize=18)

# Exibe o resultado do algoritmo UTSR.
plt.subplot(1, 3, 2)
plt.imshow(data_point.imaging_results[chave_sparsa].image, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('SpaRSA', fontsize=18)

# Exibe o resultado do algoritmo UTSR com envelope normalizado.
plt.subplot(1, 3, 3)
plt.imshow(normalize(envelope(data_point.imaging_results[chave_sparsa].image, -2)), aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('SpaRSA com pós-processamento', fontsize=18)

plt.tight_layout()
plt.show()
