import numpy as np
from matplotlib import pyplot as plt
from framework import file_omniscan
from framework.data_types import ImagingROI
from framework.post_proc import envelope, normalize
from imaging import saft, wk_saft
from imaging import bscan

shot = 0

# data = file_omniscan.read("C:/Users/LASSIP/Documents/Arquivos_teste/sdh_top_ls.opd", sel_shots=shot, pitch=0.5, dim=0.5, inter_elem=0.1,
#                 freq=5., bw=0.5, pulse_type="gaussian", n=2) #usar roi de 25
# data = file_omniscan.read("C:/Users/giova/Desktop/dados_giovanni.opd",
#                           sel_shots=shot, freq=5., bw=0.5, pulse_type="gaussian")  # usar roi de 55
# data = file_omniscan.read("C:/Users/GiovanniAlfredo/Google Drive/AUSPEX/Dados OmniScan/dados_giovanni.opd",
#                           sel_shots=shot, freq=5., bw=0.5, pulse_type="gaussian")  # usar roi de 55
# data = file_omniscan.read("C:/Users/giova/Desktop/dados_giovanni.opd",
#                           sel_shots=shot, freq=5., bw=0.5, pulse_type="gaussian")  # usar roi de 55
# data = file_omniscan.read("C:/Users/asros/Documents/dados omniscan/dados_giovanni.opd", sel_shots=shot, freq=5., bw=0.5,
#                          pulse_type="gaussian")  # usar roi de 55
data = file_omniscan.read("C:/Users/asros/Documents/dados omniscan/sdh_top_ls.opd", sel_shots=shot, freq=5., bw=0.5,
                          pulse_type="gaussian")  # usar roi de 55
# data = file_omniscan.read("C:/Users/LASSIP/Documents/Arquivos_teste/dados_giovanni.opd", sel_shots=shot, pitch=0.5, dim=0.5, inter_elem=0.1,
#                 freq=5., bw=0.5, pulse_type="gaussian", n=1) #usar roi de 55
# data = file_omniscan.read("C:/Users/LASSIP/Documents/Arquivos_teste/linear_gen_32_2.opd", sel_shots=shot, pitch=0.5, dim=0.5, inter_elem=0.1,
#                 freq=5., bw=0.5, pulse_type="gaussian", n=2) #usar roi de 55

# data = file_civa.read("C:/Users/LASSIP/Documents/Arquivos_teste/teste03_28-07-2018_setorial.opd")

# escolher qual shot exibir entre 0 e 600

corner_roi = np.array([-20, 0, 10])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=15.0, width=40.0, h_len=200, w_len=400)

# ========== B-Scan ==========
chave1 = bscan.bscan_kernel(data, roi=roi, sel_shot=shot, c=data.specimen_params.cl)
image_out_bscan = normalize(envelope(data.imaging_results[chave1].image, -2))

fig = plt.figure()
ax1 = fig.add_subplot(131)
im_bscan = ax1.imshow(image_out_bscan, aspect='auto', cmap=plt.get_cmap('Greys'),
                      extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("B-SCAN", {'fontsize': 8})

# ========== SAFT ==========
chave2 = saft.saft_kernel(data, roi=roi, sel_shot=shot, c=data.specimen_params.cl)
image_out_saft = normalize(envelope(data.imaging_results[chave2].image, -2))

ax2 = fig.add_subplot(132)
im_saft = ax2.imshow(image_out_saft, aspect='auto', cmap=plt.get_cmap('Greys'),
                     extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("SAFT", {'fontsize': 8})

# ========== wk-SAFT ==========
chave3 = wk_saft.wk_saft_kernel(data, roi=roi, sel_shot=shot, c=data.specimen_params.cl)
image_out_wk_saft = normalize(envelope(data.imaging_results[chave3].image, -2))

ax3 = fig.add_subplot(133)
im_wk_saft = ax3.imshow(image_out_wk_saft, aspect='auto', cmap=plt.get_cmap('Greys'),
                        extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("wk-SAFT", {'fontsize': 8})
plt.show()

# save_data.write_file(filename='teste2.h5', path="C:/Users/asros/Documents/dados omniscan/sdh_top_ls.opd", dados=data, roi=roi,
#                      img_bscan=chave1, img_saft=chave2, img_wk_saft=chave3)

