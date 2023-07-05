import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from framework import file_m2k, post_proc
from matplotlib.animation import FFMpegWriter

def das(echoes_all, angles, elem_pos_m, ts_s, c):
    result = np.zeros((echoes_all.shape[0], echoes_all.shape[1]))
    for shot in range(angles.shape[0]):
        echoes = echoes_all[:, shot, :]
        echoes_shifted = np.zeros_like(echoes)

        roll = np.zeros_like(elem_pos_m)
        for i in range(echoes_all.shape[2]):
            angle = angles[shot]
            angle_rad = angle * np.pi / 180
            roll[i] = np.round(elem_pos_m[i] * np.sin(angle_rad) / (c * ts_s))
            echoes_shifted[:, i] = np.roll(echoes[:, i], int(roll[i]))
        ascan = np.sum(echoes_shifted, 1)
        result[:, shot] = ascan
    return result

# file = ('ensaio_frio_falha1.m2k', 'ensaio_aquecido0a10_falha1.m2k', 'ensaio_aquecido10a30_falha1.m2k',
#         'ensaio_aquecido30a50_falha1.m2k','ensaio_aquecido50a70_falha1.m2k')

file = ('ensaio_120aquecido0a10_falha1.m2k')#, 'ensaio_120aquecido10a30_falha1.m2k',
        # 'ensaio_120aquecido30a50_falha1.m2k','ensaio_120aquecido50a70_falha1.m2k')

path = 'F:/Dados_CRAS/'
name = 0
for j in range(2):
    # if j == 0:
    #     n_shots = 1
    # else:
    n_shots = 40
    for i in range(n_shots):
        data = file_m2k.read(path + file[j], freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=i)
        echoes_all = data[0].ascan_data[:, :, :32, 0]
        angles = 180 - data[0].inspection_params.angles[:] - 55
        elem_pos_m = data[0].probe_params.elem_center[32:32+32, 0]*1e-3
        ts_s = data[0].inspection_params.sample_time * 1e-6
        c = data[0].specimen_params.cs+2000
        img = das(echoes_all, angles, elem_pos_m, ts_s, c)
        np.save(path + f'img{name}.npy', img)
        name += 1

# ## Gera video
metadata = dict(title='CRAS Test', artist='Matplotlib',
                comment='fev/2022')
writer = FFMpegWriter(fps=1, metadata=metadata)

fig, (axA, axB) = plt.subplots(1, 2)
# ax = fig.add_subplot(111)
axA.set_aspect('auto')
axB.set_aspect('auto')
list = np.linspace(0,160,161) ##np.concatenate((np.arange(0,85),np.arange(121,161)))
with writer.saving(fig, path + "writer_test.mp4", 200):
    img_ref_all = np.load(f'C:/Users/panth/Desktop/CRAS/image70/img0.npy')
    img_ref = img_ref_all[6000:12000]
    max_ref = np.max(np.log10(np.abs(img_ref) + 1e-3))
    axA.imshow(np.log10(post_proc.envelope(img_ref, axis=1) + 1e-3), aspect='auto', vmax=max_ref, animated=True)
    axA.set_title(f'Referência - Temp. Ambiente')

    for i in range(24):
        img_all = np.load(path + f'img{int(list[i*5])}.npy')
        img = img_all[6000:12000]
        max = np.max(np.log10(np.abs(img) + 1e-3))
        axB.imshow(np.log10(post_proc.envelope(img, axis=1) + 1e-3), aspect='auto', vmax=max, animated=True)
        axB.set_title(f'Aquecendo - Shot {list[i*5]}')
        writer.grab_frame()