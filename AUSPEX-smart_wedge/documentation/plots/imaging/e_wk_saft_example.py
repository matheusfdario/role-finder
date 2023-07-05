import numpy as np
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from imaging import e_wk_saft, bscan
from framework.post_proc import envelope, normalize

# --- Dados ---
# Carrega os dados de inspeção do arquivo de simulação do CIVA.
data = file_civa.read("Furo40mmPA_FMC_Contact_new.civa")

# --- ROI ---
# Define uma ROI de 20 mm x 20 mm.
height = 20.0
width = 20.0

# Define a ROI, iniciando em (-10, 0, 30) e com as dimensões definidas acima.
corner_roi = np.array([[-10.0, 0.0, 30.0]])
roi = data_types.ImagingROI(corner_roi, height=height, width=width)

# --- Processamento ---
# Obtém a imagem B-scan. Note que o algoritmo retorna apenas a chave de
# identificação, sendo que o resultado é salvo na própria variável "data".
# Além disso, o algoritmo obtém a imagem na ROI definida acima.
bscan_key = bscan.bscan_kernel(data, roi)

# Obtém a reconstrução da imagem na ROI com o algoritmo E-wk-SAFT. 
e_wk_saft_key = e_wk_saft.e_wk_saft_kernel(data, roi)

# --- Imagens ---
plt.figure(figsize=(16, 7))

# Exibe o resultado do algoritmo B-scan. 
plt.subplot(1, 3, 1)
plt.imshow(data.imaging_results[bscan_key].image, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('B-scan', fontsize=18)

# Exibe o resultado do algoritmo E-wk-SAFT.
plt.subplot(1, 3, 2)
plt.imshow(data.imaging_results[e_wk_saft_key].image, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title(r'E-$\omega k$-SAFT', fontsize=18)

# Exibe o resultado do algoritmo E-wk-SAFT com envelope normalizado.
plt.subplot(1, 3, 3)
plt.imshow(normalize(envelope(data.imaging_results[e_wk_saft_key].image, 0)), aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title(r'E-$\omega k$-SAFT com pós-processamento', fontsize=18)

plt.tight_layout()
plt.show()
