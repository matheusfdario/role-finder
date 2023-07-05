import numpy as np
from framework import file_m2k
import numpy as np
import matplotlib.pyplot as plt


root = "/media/tekalid/Data/CENPES/junho_2022/"
filename = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1"
path = root + filename

i = 0
# O número de shots do arquivo de referência são 78.
N_shots = 78
for i in range(N_shots):
    print('Loading ' + str(i))
    data_insp = file_m2k.read(path + '.m2k', sel_shots=[i], type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5, tp_transd='gaussian')[0]
    if i == 0:
        avg = np.zeros_like(data_insp.ascan_data_sum, dtype=float)
    avg += data_insp.ascan_data_sum
    print('ok')
    i += 1
avg /= N_shots
# plt.imshow(np.sum(avg, 2), aspect='auto', interpolation='none')
np.save('media_refencia_sum.npy', avg)