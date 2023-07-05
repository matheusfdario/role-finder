import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k
import os
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import (LinearLocator, FormatStrFormatter,
                               AutoMinorLocator)
from framework.post_proc import envelope
from generate_law import *

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def GenerateImage(shot):
    # Lê o arquivo a ser reconstruido:

    extension = ".m2k"
    path = data_root + filename + extension
    data = file_m2k.read(path, sel_shots=shot, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian')
    data_4db = data[0]
    data_15db = data[1]

    # Lê o arquivo de referência das médias:
    smartwedge_mask = np.load("media_referencia.npy")

    # Tempo de ensaio em micro segundos:
    t_span = np.linspace(start=data[0].inspection_params.gate_start,
                         stop=data[0].inspection_params.gate_end,
                         num=data[0].inspection_params.gate_samples)

    # Calcula os tempos entre o centro do transdutor e a superfície da tubulação:
    delay_pattern = np.zeros_like(angles, dtype=float)
    time_spent = np.zeros_like(delay_pattern)
    wedge_time = np.zeros_like(delay_pattern)
    for i, bet in enumerate(betas):
        delay_pattern[i] = compensate_time(bet, sm) * 2
        time_spent[i] = delay_pattern[i] / 1e-6
        wedge_time[i] = compensate_time_wedge(bet, sm)/1e-6 * 2
    delay_pattern = delay_pattern - delay_pattern.min()
    delay_pattern = delay_pattern / 1e-6

    nonzero = delay_pattern[0]
    for i in np.arange(0, delay_pattern.shape[0]):
        if delay_pattern[i] < 1e-6:
            delay_pattern[i] = nonzero
        else:
            delay_pattern[i] = 0

    # Transforma os atrasos em indices a deslocar as linhas:
    sample_time = data[0].inspection_params.sample_time
    # Define limites de saturação:
    log_offset = 1e2



    img_ref = np.sum(smartwedge_mask, 2)
    img_ref = np.log10(np.abs(img_ref) + log_offset)


    # Plotando a figura:
    plt.suptitle(f"Shot : {shot} || Ensaio: {filename}")

    plt.subplot(2,2,1)
    # Imagem do Ensaio a 4 dB
    plt.subplot(2, 2, 1)
    img_insp_4db = np.sum(data_4db.ascan_data, 2)
    img_insp_4db = np.log10(np.abs(img_insp_4db) + log_offset)

    vmin_cte = img_insp_4db.min()
    vmax_cte = img_insp_4db.max()

    plt.imshow(img_insp_4db, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Ganho = 4 dB")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext.')
    plt.plot(betas, time_spent+ + 2*(16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5, label='Sup. Int.')
    vetor_medio = np.array([moving_average(img_insp_4db[:, i, 0], 7) for i in range(img_insp_4db.shape[1])]).transpose()
    max_idx = np.argmax(vetor_medio, 0) + 1
    plt.plot(betas, np.array([t_span[i] for i in max_idx]),
             'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")


    # Imagem do Ensaio a 15 dB
    plt.subplot(2, 2, 2)
    img_insp_15db = np.sum(data_15db.ascan_data, 2)
    img_insp_15db = np.log10(np.abs(img_insp_15db) + log_offset)

    plt.imshow(img_insp_15db, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Ganho = 15 dB")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext.')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int.')


    max_idx = np.argmax(img_insp_15db, 0)
    plt.plot(betas, np.array([t_span[i] for i in max_idx]),
             'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")


    # Imagem do Ensaio a corrigida
    plt.subplot(2, 2, 3)
    cte_4db = 10**(4/20)
    cte_7db = 10**(7/20)
    cte_15db = 10**(15/20)
    img_insp_4db_sub = np.sum(data_4db.ascan_data - smartwedge_mask * cte_4db / cte_7db, 2)
    img_insp_4db_sub = np.log10(np.abs(img_insp_4db_sub) + log_offset)
    img_insp_15db_sub = np.sum(data_15db.ascan_data - smartwedge_mask * cte_15db / cte_7db, 2)
    img_insp_15db_sub = np.log10(np.abs(img_insp_15db_sub) + log_offset)

    img_combined = np.zeros_like(img_insp_4db_sub)
    for i, delay in enumerate(delay_pattern):
        if delay == 0:
            img_combined[:, i, 0] = img_insp_15db_sub[:, i, 0]
        else:
            img_combined[:, i, 0] = img_insp_4db_sub[:, i, 0]

    vmin_cte = img_combined.min()
    vmax_cte = img_combined.max()
    plt.imshow(img_combined, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Imagem Combinada e Subtraida")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')

    plt.plot(betas, time_spent, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext.')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int.')

    max_idx = np.argmax(img_combined, 0)
    plt.plot(betas, np.array([t_span[i] for i in max_idx]),
             'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")

    # Imagem do Ensaio a corrigida dB
    plt.subplot(2, 2, 4)
    img_shifted = shiftImage(img_combined[:, :, 0], delay_pattern, sample_time)

    plt.imshow(img_shifted, aspect='auto', interpolation='bicubic', extent=[-92.5, 92.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    plt.xlabel("Ângulo de Varredura")
    plt.ylabel("Tempo em $\mu$s")
    plt.title(f"Imagem Combinada, Subtraida e Deslocada (Com Interpolação)")
    ax = plt.gca()
    ax.xaxis.grid(alpha=0, which="major", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.set_xticks(np.arange(-90, 90 + step * 3, step * 3), minor=False)
    ax.set_xticks(np.arange(-90 - step / 2, 90 + step / 2 + step, step), minor=True)
    ax.set_ylim([90, 40])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='minor',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
    )
    alpha_cte = .5
    plt.plot(betas, wedge_time, 'x', color=[1, 0, 1], alpha=alpha_cte, markersize=3, label='Pos. Sapata')
    plt.plot(betas, time_spent + delay_pattern, 'd', color=[1, 0, 0], alpha=alpha_cte, markersize=1.5, label='Sup. Ext.')
    plt.plot(betas, time_spent + + 2 * (16e-3) / (6.3e3) * 1e6 + delay_pattern, 'd', color=[.5, 0, .5], alpha=alpha_cte, markersize=1.5,
             label='Sup. Int.')

    max_idx = np.argmax(img_shifted, 0)
    plt.plot(betas, np.array([t_span[i] for i in max_idx]),
             'o', color=[0, 1, 0], alpha=alpha_cte - .2, markersize=3, label="Máx. Col.")

    plt.legend(loc="upper center")

    plt.tight_layout()

    return None

# Importa módulos associados à smartwedge_old:
from generate_law import *

# GEra um Bscan onde
# Eixo X são os elementos e Eixo Y são os tempos. Varia-se o shot (ângulo fixo).

# Dados da sapata do Guastavo:
pitch = 0.6e-3
coord = np.zeros((64, 2))
coord[:, 0] = np.linspace(-63 * pitch / 2, 63 * pitch / 2, 64)

# Ellipse parameters:
c = 84.28
r0 = 67.15
wc = 6.2
offset = 2
Tprime = 19.5

# Velocidade do som nos materiais em mm/us
v1 = 6.37
v2 = 1.43

# Criação do objeto smartwedge_old:
sm = smartwedge(c, r0, wc, v1, v2, Tprime, offset)

# Define ângulos de varredura no referencial da tubulação:
beg_angle = -90
end_angle = 90
step = 5
betas = np.arange(beg_angle, end_angle + step, step)

# Calcula os ângulos no referncial do transdutor (ângulo de disparo da onda plana):
angles = list()
for beta in betas:
    ang, _ = sm.compute_entrypoints(beta)
    angles.append(ang)
angles = np.rad2deg(np.array(angles))

# Define o caminho dos dados:
data_root = "C:/Users/Thiago/repos/Dados/AUSPEX/CENPES/jun2022/"
project_root = "C:/Users/Thiago/repos/AUSPEX_new/test/smartwedge_old"
filename = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1"
path1 = "resultados"
result_foldername = path1 + "/" + filename
extension = ".m2k"
data_path = data_root + filename + extension

data = file_m2k.read(data_path, sel_shots=0, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                     tp_transd='gaussian')



# Parâmetros do Vídeo:
generate_video = False

# O número de shots:
n_shot = 97


video_title = "VistaSetorial_Combinada_ComReferência"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)
fig = plt.figure(figsize=(14, 9))
# fig = plt.figure(constrained_layout=True, figsize=(14, 9))

shot = 0
if generate_video == True:
    with writer.saving(fig, result_foldername + "/" + video_title + "14dB_" + ".mp4", dpi=300):
        for shot in range(n_shot):
            GenerateImage(shot)
            print(f"frame = {shot + 1}")
            writer.grab_frame()
            plt.clf()
else:
    GenerateImage(shot)