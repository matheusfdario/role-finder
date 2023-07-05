import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from framework import file_m2k, post_proc
from matplotlib.animation import FFMpegWriter



    for i in range(n_shots):
        data = file_m2k.read(path + file[j], freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=i)
       ##  TFM
	img = tfm....
        np.save(f'img{name}.npy', img)
        name += 1

# ## Gera video
metadata = dict(title='CRAS Test', artist='Matplotlib',
                comment='fev/2022')
writer = FFMpegWriter(fps=1, metadata=metadata)

fig, ax = plt.figure()
ax.set_aspect('auto')
list = np.arange(n_shots) 
with writer.saving(fig, path + "writer_test.mp4", 200):
    for i in range(n_shots):
        img_all = np.load(f'img{int(list[i*5])}.npy')
        max = np.max(np.log10(np.abs(img) + 1e-3))
        axB.imshow(np.log10(post_proc.envelope(img, axis=1) + 1e-3), aspect='auto', vmax=max, animated=True)
        axB.set_title(f'Titulo - Shot {list[i*5]}')
        writer.grab_frame()