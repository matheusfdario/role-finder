from framework import file_civa
import numpy as np
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from imaging.tfm import tfm_kernel, ImagingROI

# Script faz a análise usando API do ensaio deslocando o furo para margem da imagem.

def crop_ascan(ascan, t_span, t0=None, tf=None):
    if t0 is not None and tf is not None:
        t0_idx = np.argmin(np.power(t_span - t0, 2))
        tf_idx = np.argmin(np.power(t_span - tf, 2))
        return ascan[t0_idx:tf_idx, :]


simulations_root = "/media/tekalid/Data/smartwedge_data/06-28-2023/"
original_root = "fmc-smartwedge_original-res.civa"  # Diretório da smartwedge original
circle_root = "fmc-ponta_adocada-res.civa"  # Diretório da smartwedge com ponta adoçada redonda
square_root = "fmc-ponta_adocada_quadrada-res.civa"  # Diretório da smartwedge com ponta adoçada quadrada

betas = np.arange(-40, 40+0.5, .5)

data_original = file_civa.read(simulations_root + original_root, sel_shots=0)
data_circle = file_civa.read(simulations_root + circle_root, sel_shots=0)
data_square = file_civa.read(simulations_root + square_root, sel_shots=0)

t_span_original = data_original.time_grid

# Corta o scan e timegrid para range desejado:
t0 = 0
tend = 100

#
log_cte = .1
nElement = 80
xres = 4
zres = 4

# Altura e largura da smartwedge quadrada (que é a maior):
width = 2 * 83.14 + 40
height = 120.62 + 60


# Cria ROI:
corner_roi = np.zeros((1, 3))
corner_roi[0] = [-width/2, 0, 0]
roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height*zres), w_len=int(width*xres))

print("TFM 1")
tfm_original = data_original.imaging_results[tfm_kernel(data_original, roi=roi, c=data_original.specimen_params.cl)].image

print("TFM 2")
tfm_circle = data_circle.imaging_results[tfm_kernel(data_circle, roi=roi, c=data_circle.specimen_params.cl)].image

print("TFM 3")
tfm_square = data_square.imaging_results[tfm_kernel(data_square, roi=roi, c=data_square.specimen_params.cl)].image

#
xmin = -width/2
xmax = width/2
zmin = corner_roi[0][2]
zmax = height + corner_roi[0][2]


# Faz o TFM:
tfm_original = np.log10(envelope(tfm_original, axis=0) + log_cte)
tfm_circle = np.log10(envelope(tfm_circle, axis=0) + log_cte)
tfm_square = np.log10(envelope(tfm_square, axis=0) + log_cte)

#
vmin_tfm = np.min(tfm_original)
vmax_tfm = (np.max(tfm_original) + np.min(tfm_original))/2
# vmax_tfm = np.max(tfm_original)

# TFM dos resultados:
plt.figure(figsize=(12,6))
plt.suptitle("TFM para três modelos diferentes de smartwedge com varredura de 90 graus.")
plt.subplot(1, 3, 1)
plt.title("Smartwedge original")
plt.imshow(tfm_original, extent=[xmin, xmax, zmax, zmin], aspect='equal', interpolation="None",
           vmin=vmin_tfm, vmax=vmax_tfm, cmap="magma")
plt.xlabel("Eixo z em mm")
plt.ylabel(r"Eixo x em mm")

plt.subplot(1, 3, 2)
plt.title("Original com pontas adoçadas")
plt.imshow(tfm_circle, extent=[xmin, xmax, zmax, zmin], aspect='equal', interpolation="None",
           vmin=vmin_tfm, vmax=vmax_tfm, cmap="magma")

plt.subplot(1, 3, 3)
plt.title("Quadrada com pontas adoçadas")
plt.imshow(tfm_square, extent=[xmin, xmax, zmax, zmin], aspect='equal', interpolation="None",
           vmin=vmin_tfm, vmax=vmax_tfm, cmap="magma")

plt.tight_layout()
