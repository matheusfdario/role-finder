import numpy as np
import ttictoc
from matplotlib import pyplot as plt
from sys import platform

from framework import file_civa, file_m2k
from framework.data_types import ImagingROI
from framework.post_proc import envelope, normalize, api
from imaging import saft, tfm, wk_saft, wavenumber

# data = file_mat.read("../data/DadosEnsaio.mat")

if platform == "win32":
    # Caminhos para arquivos com simulações no Windows.
    # data = file_mat.read("../data/DadosEnsaio.mat")
    # data = file_m2k.read("C:/Users/CIVA/Documents/Arquivos M2K/CP1_Bot_50_Direct.m2k", sel_shots=0)
    # data = file_m2k.read("C:/Users/GiovanniAlfredo/Documents/Projetos/AUSPEX/Data/CP1_Bot_50_Direct.m2k", sel_shots=0)
    # data = file_civa.read("C:/Users/CIVA/Documents/Arquivos CIVA/Furo40mmPA_FMC_Contact_new.civa")
    data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
    # data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Furo40mmPA_FMC_Contact_Scanning.civa")
    # data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Testing01_ImpactPoint_waterpath10mm.civa")
    # data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Testing01_CrystalCenter_airpath5mm.civa")
    # data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Testing01_CrystalCenter_waterpath5mm.civa")
    # data = file_civa.read("C:/Users/GiovanniAlfredo/Desktop/CompartVM/Testing02_Filipe_step35.civa")
else:
    # Caminhos para arquivos com simulações no Linux.
    data = file_civa.read("/home/giovanni/Documents/CompartVM/Furo40mmPA_FMC_Contact_new.civa")
    # data = file_m2k.read("/home/giovanni/Documents/ArquivosM2K/CP1_Bot_50_Direct.m2k", sel_shots=0)

cl = 5900.0
# cl = 6300.0

# Implementa primeiramente a versão w-k SAFT (wavenumber mono-elemento - Stepinski2007).
# Define a ROI
corner_roi = np.array([-10.0, 0.0, 30.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=20.0, width=20.0, h_len=200, w_len=200)
# corner_roi = np.array([data.probe_params.elem_center[0, 0],
#                        0.0,
#                        30.0])[np.newaxis, :]
# idx_t0 = np.searchsorted(data.time_grid[:, 0], (2*30e-3/cl) / 1e-6)
# idx_te = np.searchsorted(data.time_grid[:, 0], (2*50e-3/cl) / 1e-6)
# roi = ImagingROI(corner_roi,
#                  height=20.0,
#                  width=data.probe_params.elem_center[-1, 0] - data.probe_params.elem_center[0, 0] + data.probe_params.pitch,
#                  h_len=(idx_te - idx_t0),
#                  w_len=float(data.probe_params.num_elem))

sel_shot = 0
wavelength = cl / (data.probe_params.central_freq * 1e6)
t = ttictoc.TicToc()

# Executa SAFT
t.tic()
chave_saft = saft.saft_kernel(data, roi=roi, c=cl)
t.toc()
print("SAFT executado em %f segundos" % t.elapsed)
image_out_saft = normalize(envelope(data.imaging_results[chave_saft].image, -2))
api_saft = api(image=image_out_saft, roi=data.imaging_results[chave_saft].roi, wavelength=wavelength)

# Executa wk-SAFT (Stepinsky2007)
t.tic()
chave_wk_saft = wk_saft.wk_saft_kernel(data, roi=roi, c=cl)
t.toc()
print("wk-SAFT executado em %f segundos" % t.elapsed)
image_out_wk_saft = normalize(envelope(data.imaging_results[chave_wk_saft].image, -2))
api_wk_saft = api(image=image_out_wk_saft, roi=data.imaging_results[chave_wk_saft].roi, wavelength=wavelength)

# Executa TFM
t.tic()
chave_tfm = tfm.tfm_kernel(data, roi=roi, c=cl)
t.toc()
print("TFM executado em %f segundos" % t.elapsed)
image_out_tfm = normalize(envelope(data.imaging_results[chave_tfm].image, -2))
api_tfm = api(image=image_out_tfm, roi=data.imaging_results[chave_tfm].roi, wavelength=wavelength)

# Executa WAVENUMBER (Hunter2008)
t.tic()
chave_wavenumber = wavenumber.wavenumber_kernel(data, roi=roi, c=cl)
t.toc()
print("WAVENUMBER executado em %f segundos" % t.elapsed)
image_out_wavenumber = normalize(envelope(data.imaging_results[chave_wavenumber].image, -2))
api_wavenumber = api(image=image_out_wavenumber, roi=data.imaging_results[chave_wavenumber].roi, wavelength=wavelength)

# Plota as imagens
cmap_string = 'jet'
# cmap_string = 'Greys'
fig = plt.figure()
ax1 = fig.add_subplot(221)
im_saft = ax1.imshow(image_out_saft, aspect='auto', cmap=plt.get_cmap(cmap_string),
                     extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("SAFT (API = %.3f)" % api_saft, {'fontsize': 8})

ax2 = fig.add_subplot(222)
im_wk_saft = ax2.imshow(image_out_wk_saft, aspect='auto', cmap=plt.get_cmap(cmap_string),
                        extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("wk-SAFT (API = %.3f)" % api_wk_saft, {'fontsize': 8})

ax3 = fig.add_subplot(223)
im_tfm = ax3.imshow(image_out_tfm, aspect='auto', cmap=plt.get_cmap(cmap_string),
                    extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("TFM (API = %.3f)" % api_tfm, {'fontsize': 8})

ax4 = fig.add_subplot(224)
im_wavenumber = ax4.imshow(image_out_wavenumber, aspect='auto', cmap=plt.get_cmap(cmap_string),
                           extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("WAVENUMBER (API = %.3f)" % api_wavenumber, {'fontsize': 8})

plt.show()
