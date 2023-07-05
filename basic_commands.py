#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 2023

@author: tatiprado
"""

"1 - Branchs"

"2 - Leitura de dados do simulador (*.civa)" \
"Lembrete: ao fazer as simulações salvar peaks+channels"
from framework import file_civa
path = 'D:/Downloads/toolbox/pc_wilcox_fmc.civa'
data = file_civa.read(path, sel_shots=None, read_ascan=True)
# "[tempo,emissor,receptor,shots]"
# "O número de shots é determinado pelas configurações do encoder/varredura"

"3 - Leitura de dados do Panther (*.m2k ou *.data_acquire)"
from framework import file_m2k
path = 'D:/Downloads/toolbox/pc_wilcox_fmc.m2k'
data = file_m2k.read(path, freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=3, read_ascan=True, type_insp="contact",
         water_path=0.0)

"A-scan"
import matplotlib.pyplot  as plt
plt.figure(1)
plt.plot(data.time_grid, data.ascan_data[:,0,0,0], label='Sinal')
plt.xlabel('Tempo [$\mu s$]')
plt.ylabel('Amplitude')
plt.title('A-scan')

# Pós-processamento
from framework import post_proc
envelope = post_proc.envelope(data.ascan_data[:,0,0,0], axis=0)
plt.plot(data.time_grid, envelope, label='Envoltória')
plt.legend()
plt.show()


"B-scan"
from imaging import bscan
import numpy as np
from framework.data_types import ImagingROI

# Parâmetros da ROI
corner_roi = np.array([-20.0, 0.0, 0])[np.newaxis, :]  #[x0, y0, z0]
roi = ImagingROI(corner_roi, height=60.0, width=40.0, h_len=10*60, w_len=10*40)

# Calculo do B-scan
chave = bscan.bscan_kernel(data, roi=roi, output_key=1, description="", sel_shot=0, c=None)

# Plot
plt.figure(2)
plt.imshow(data.imaging_results[chave].image, aspect='auto')
plt.colorbar()
plt.ylabel('Tempo [$\mu s$]')
plt.xlabel('Elemento - emissor=receptor')
plt.title('B-scan')
plt.show()

# Raw processing
# image_B = np.diagonal(data.ascan_data[:,:,:,0], axis1=1, axis2=2)

"SAFT"
from imaging import saft

# Parâmetros da ROI
corner_roi = np.array([-20.0, 0.0, 0])[np.newaxis, :]  #[x0, y0, z0]
roi = ImagingROI(corner_roi, height=60.0, width=40.0, h_len=10*60, w_len=10*40)

# Calculo do SAFT
chave = saft.saft_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl)

# Plot
plt.figure(3)
plt.imshow(data.imaging_results[chave].image, aspect='auto')
plt.title('SAFT')
plt.show()

plt.figure(4)
plt.imshow(data.imaging_results[chave].image, aspect='auto', extent=[-20,20,60,0])
plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.title('SAFT')
plt.show()

"TFM"
from imaging import tfm
# Parâmetros da ROI
corner_roi = np.array([-20.0, 0.0, 15])[np.newaxis, :]  #[x0, y0, z0]
roi = ImagingROI(corner_roi, height=15.0, width=40.0, h_len=10*15, w_len=10*40)
# Brinde - Facilita o uso do extent
x_inf = corner_roi.min()
y_sup = corner_roi.max()
x_sup = x_inf + roi.width
y_inf = y_sup + roi.height

# Calculo do TFM
chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0, c=data.specimen_params.cl)
tfm_o = post_proc.envelope(data.imaging_results[chave].image)

# Plot
plt.figure(5)
plt.imshow(tfm_o, aspect='auto', extent=[x_inf, x_sup, y_inf, y_sup])
plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.title('TFM')
plt.show()

plt.figure(6)
plt.imshow(post_proc.normalize(tfm_o), aspect='auto', extent=[x_inf, x_sup, y_inf, y_sup])
plt.colorbar()
plt.xlabel('x [mm]')
plt.ylabel('z [mm]')
plt.title('TFM Normalizado')
plt.show()


"CPWC"
from imaging import cpwc
path = 'D:/Downloads/toolbox/pc_1furo_pwi50.civa'
data_pwi = file_civa.read(path)

# Tamanho da ROI
height = 50.0
width = 40.0

# Define a ROI
corner_roi = np.array([[-20, 0.0, 10.0]])
roi = ImagingROI(corner_roi, height=height, width=width,
                            h_len=100, w_len=100)

data_pwi.ascan_data = data_pwi.ascan_data_sum
chave = cpwc.cpwc_kernel(data_pwi, roi, angles=data_pwi.inspection_params.angles)
ic = data.imaging_results[chave].image

# --- Output ---
plt.figure(7)
plt.imshow(post_proc.normalize(post_proc.envelope(ic, -2)), aspect='auto',
           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("CPWC")

