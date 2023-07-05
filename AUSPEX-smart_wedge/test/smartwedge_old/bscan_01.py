import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k
import os
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import (LinearLocator, FormatStrFormatter,
                               AutoMinorLocator)
from framework.post_proc import envelope


def GenerateImage(shot):
    # Lê o arquivo a ser reconstruido:

    extension = ".m2k"
    path = data_root + filename + extension
    data = file_m2k.read(path, sel_shots=shot, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                         tp_transd='gaussian')
    img_ascan = data[0].ascan_data

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


    img_mask = smartwedge_mask

    # Define limites de saturação:
    log_offset = 1e-3


    # Plotando a figura:
    plt.suptitle(f"Shot : {shot} || Ensaio: {filename}")

    # Ângulo 1:
    f_ax1 = fig.add_subplot(gs[0, 0])
    idx = 0
    beta1 = betas[idx]
    img = img_ascan[:, idx, :]
    # img = (img - img.min())/(img.max() - img.min())
    img = np.log10(np.abs(img) + log_offset)


    vmin_cte = 0
    vmax_cte = img.max()

    plt.imshow(img, aspect='auto', interpolation='none', extent=[-0.5, 63.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    f_ax1.set_xlabel("Elementos")
    f_ax1.set_ylabel("Tempo em $\mu$s")
    plt.title(f"Ângulo de Varredura de {beta1:.0f}°")
    ax = plt.gca()
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="major", color=[0, 0, 0])
    ax.set_xticks(np.arange(-0.5, 63.5, 1), minor=True)
    ax.set_xticks(np.arange(-0.5, 63.5, 9), minor=False)

    # Ângulo 2:
    f_ax2 = fig.add_subplot(gs[0, 1])
    idx = 18
    beta1 = betas[idx]
    img = img_ascan[:, idx, :]
    # img = (img - img.min()) / (img.max() - img.min())
    img = np.log10(np.abs(img) + log_offset)

    plt.imshow(img, aspect='auto', interpolation='none', extent=[-0.5, 63.5, t_span[-1], t_span[0]],
               vmin=vmin_cte, vmax=vmax_cte, cmap=plt.get_cmap("Greys").reversed())
    f_ax2.set_xlabel("Elementos")
    f_ax2.set_ylabel("Tempo em $\mu$s")
    plt.title(f"Ângulo de Varredura de {beta1:.0f}°")
    ax = plt.gca()
    ax.xaxis.grid(alpha=.05, which="minor", color=[0, 0, 0])
    ax.xaxis.grid(alpha=.05, which="major", color=[0, 0, 0])
    ax.set_xticks(np.arange(-0.5, 63.5, 1), minor=True)
    ax.set_xticks(np.arange(-0.5, 63.5, 9), minor=False)

    f_ax3 = fig.add_subplot(gs[1, :])
    img = np.sum(img_ascan, 2)
    # img = (img - img.min())/(img.max() - img.min())
    img = np.log10(np.abs(img) + 1e1)
    plt.imshow(img, aspect='auto', interpolation='none', extent=[-92.5, 92.5, t_span[-1], t_span[0]], vmin=img.min(),
               vmax=img.max(), cmap=plt.get_cmap("Greys").reversed())
    plt.title("Inspeção")

    # Plotando curvas que representam a distância teórica para cada ângulo:
    direct_time = np.ones_like(betas, dtype=float) * compensate_time(0, sm) / 1e-6 * 2
    indirect_time = np.ones_like(betas, dtype=float) * compensate_time(60, sm) / 1e-6 * 2

    # Plotando máximo por coluna:
    max_row_idx = np.argmax(img, 0)
    max_time = np.array([t_span[i] for i in max_row_idx])
    plt.plot(betas, time_spent, 'x', color=[0, 0, 1], alpha=0.4, markersize=1.2, label='Sup. Externa Tubulação')
    plt.plot(betas, max_time, 'o', color=[1, 0, 0], markersize=0.9, alpha=0.4, label='Máximo por Coluna')
    plt.plot(betas, wedge_time, 'o', color=[1, 0, 1], markersize=0.9, alpha=0.4, label='Sapata')
    # PLotando a grade:
    ax = plt.gca()
    ax.set_ylim([90, 40])
    ax.xaxis.grid(alpha=.05, which="minor", color="black")
    ax.xaxis.grid(alpha=.05, which="major", color="black")
    ax.set_xticks(np.arange(-92.5, 92.5, 5), minor=True)
    ax.set_xticks(np.arange(-92.5, 92.5, 15), minor=False)
    plt.legend(loc='upper right')
    ax.set_xlabel("Ângulo de Varredura")
    ax.set_ylabel("Tempo em $\mu$s")

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
step = 2
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
filename = "Smart Wedge 29-06-22 100 V Sem Foco Step 2 Aquisicao 0h v1"
path1 = "resultados"
result_foldername = path1 + "/" + filename
extension = ".m2k"
data_path = data_root + filename + extension

data = file_m2k.read(data_path, sel_shots=0, type_insp='contact', water_path=0, freq_transd=5, bw_transd=0.5,
                     tp_transd='gaussian')
# data[0].ascan_data.min()
# data[0].ascan_data.min()

# plt.imshow(data[0].imshow())
# plt.imshow()


# Parâmetros do Vídeo:
generate_video = False

# O número de shots:
n_shot = 33



video_title = "Bscan_ao_longo_dos_Shots_Sem_Subtração"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)
# fig = plt.figure(figsize=(10, 8))
fig = plt.figure(constrained_layout=True, figsize=(14, 9))
gs = fig.add_gridspec(2, 2)

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