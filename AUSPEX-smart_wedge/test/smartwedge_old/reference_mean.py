import numpy as np
from framework import file_m2k
import numpy as np
import matplotlib.pyplot as plt

root =  "C:/Users/Thiago/repos/Dados/AUSPEX/CENPES/jun2022/"
filename = 'Smart Wedge Aquisicao de Referencia com 7 dB.m2k'
path = root + filename

i = 0
# O número de shots do arquivo de referência são 78.
N_shots = 78
for i in range(N_shots):
    print('Loading ' + str(i))
    data_insp = file_m2k.read(path, sel_shots=[i], type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
    if i == 0:
        avg = np.zeros_like(data_insp.ascan_data, dtype=float)
    avg += data_insp.ascan_data
    print('ok')
    i += 1
avg /= N_shots
# plt.imshow(np.sum(avg, 2), aspect='auto', interpolation='none')
np.save('media_refecrencia.npy', avg)