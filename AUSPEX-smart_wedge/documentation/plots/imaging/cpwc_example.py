import numpy as np
import matplotlib.pyplot as plt
from framework import data_types, file_civa
from imaging import cpwc, bscan
from framework.post_proc import envelope, normalize

# --- Dados ---
# Carrega os dados de inspeção do arquivo de simulação do CIVA.
data = file_civa.read("../../../data/peca_80_60_25_ensaio_pw_validation.civa")

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

# Obtém a reconstrução da imagem na ROI com o algoritmo CPWC. 
cpwc_key = cpwc.cpwc_kernel(data, roi)

# --- Imagens ---
plt.figure(figsize=(16, 7))

# Exibe o resultado do algoritmo B-scan. 
plt.subplot(1, 3, 1)
plt.imshow(data.imaging_results[bscan_key].image, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('B-scan', fontsize=18)

# Exibe o resultado do algoritmo CPWC.
plt.subplot(1, 3, 2)
plt.imshow(data.imaging_results[cpwc_key].image, aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('CPWC', fontsize=18)

# Exibe o resultado do algoritmo CPWC com envelope normalizado.
plt.subplot(1, 3, 3)
plt.imshow(normalize(envelope(data.imaging_results[cpwc_key].image, -2)), aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title('CPWC com pós-processamento', fontsize=18)

plt.tight_layout()
plt.show()
