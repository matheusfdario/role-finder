import numpy as np
from matplotlib import pyplot as plt

from framework import file_m2k, post_proc, file_civa
from framework.data_types import ImagingROI
from imaging import tfm, saft

data = file_m2k.read("/home/hector/PycharmProjects/AUSPEX/data/tubo_acrilico_com_dentes_FMC.m2k", freq_transd=5.0, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
# data = file_civa.read("/home/hector/PycharmProjects/AUSPEX/data/fmc_noise_hist.civa")

#corner_roi = np.array([-20.0, 12.0, 25.0])[np.newaxis, :]
#roi = ImagingROI(corner_roi, height=65.0, width=40.0, h_len=100, w_len=64)
corner_roi = np.array([-20.0, 12.0, 20.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=65.0, width=40.0, h_len=100, w_len=64)

#i = 5
#chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0)
chave = saft.saft_kernel(data, roi=roi, sel_shot=0, c=6300.0)
#while i > 0:
#    chave = tfm.tfm_kernel(data, roi=roi, output_key=chave, sel_shot=0)
#    i = i - 1

plt.imshow(abs(data.imaging_results[chave].image))
plt.show()