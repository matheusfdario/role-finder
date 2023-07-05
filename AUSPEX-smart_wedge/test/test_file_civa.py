# import numpy as np
# from matplotlib import pyplot as plt
#
# from framework import file_civa
#
# data = file_civa.read("C:/Users/CIVA/Documents/Arquivos CIVA/Furo40mmPA_FMC_Contact.civa")
#
# plt.imshow(data.ascan_data[:, 0, 0, :], aspect='auto')
# plt.show()


import numpy as np
from matplotlib import pyplot as plt
from framework.data_types import ImagingROI
from framework.post_proc import envelope, normalize
from imaging import bscan, saft, wk_saft
from framework import file_civa

data = file_civa.read("C:/Users/asros/Documents/dados civa/Furo40mmPA_FMC_Contact_new.civa")

corner_roi = np.array([-20, 0, 30])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=20.0, width=40.0, h_len=200, w_len=400)

# ========== B-Scan ==========
chave1 = bscan.bscan_kernel(data, roi=roi, sel_shot=0, c=5900.0)
image_out_bscan = normalize(envelope(data.imaging_results[chave1].image, -2))

fig = plt.figure()
ax1 = fig.add_subplot(131)
im_bscan = ax1.imshow(image_out_bscan, aspect='auto', cmap=plt.get_cmap('Greys'),
                      extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("B-SCAN", {'fontsize': 8})

# ========== SAFT ==========
chave2 = saft.saft_kernel(data, roi=roi, sel_shot=0, c=5900.0)
image_out_saft = normalize(envelope(data.imaging_results[chave2].image, -2))

ax2 = fig.add_subplot(132)
im_saft = ax2.imshow(image_out_saft, aspect='auto', cmap=plt.get_cmap('Greys'),
                     extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("SAFT", {'fontsize': 8})

# ========== wk-SAFT ==========
chave3 = wk_saft.wk_saft_kernel(data, roi=roi, sel_shot=0, c=5900.0)
image_out_wk_saft = normalize(envelope(data.imaging_results[chave3].image, -2))

ax3 = fig.add_subplot(133)
im_wk_saft = ax3.imshow(image_out_wk_saft, aspect='auto', cmap=plt.get_cmap('Greys'),
                        extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
plt.title("wk-SAFT", {'fontsize': 8})
plt.show()