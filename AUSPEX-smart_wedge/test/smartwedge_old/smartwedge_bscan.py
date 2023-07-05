import numpy as np
import matplotlib.pyplot as plt
from framework import file_m2k
import cv2

# Importa módulos associados à smartwedge_old:
from generate_law import *

def shiftImage(sectorial_img, delays, time_sample):
    shifts = np.round(delays / time_sample).astype(int)
    m, n = sectorial_img.shape
    m_new = np.round(m + np.max(shifts)).astype(int)
    shifted_img = np.zeros((m_new, n))

    for i in np.arange(0, n):
        shifted_img[shifts[i]:shifts[i] + m, i] = sectorial_img[:, i]
    shifted_img = shifted_img[:m, :n]
    return shifted_img

if __name__ == "__main__":
    # Dados da sapata do Guastavo:
    pitch = 0.6e-3
    coord = np.zeros((64, 2))
    coord[:, 0] = np.linspace(-63 * pitch/2, 63 * pitch/2, 64)

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
    betas = np.arange(beg_angle, end_angle+step, step)

    # Calcula os ângulos no referncial do transdutor (ângulo de disparo da onda plana):
    angles = list()
    for beta in betas:
        ang, _ = sm.compute_entrypoints(beta)
        angles.append(ang)
    angles = np.rad2deg(np.array(angles))

    root = "C:/Users/Thiago/repos/Dados/AUSPEX/CENPES/jun2022/"
    filename = "Smart Wedge Aquisicao 3 em 3h step 5 15-19-57.m2k"

    # Lê o arquivo:
    data = file_m2k.read(root+filename, sel_shots=50, freq_transd=5, bw_transd=0.5, tp_transd='gaussian')
    # Tempo de ensaio em micro segundos:
    t_span = np.linspace(data.inspection_params.gate_start, data.inspection_params.gate_end, data.inspection_params.gate_samples)



    # Calcula os tempos entre o centro do transdutor e a superfície da tubulação:
    delay_pattern = np.zeros_like(angles)
    for i, bet in enumerate(betas):
        delay_pattern[i] = compensate_time(bet, sm)
    delay_pattern = delay_pattern - delay_pattern.min()
    delay_pattern = delay_pattern/1e-6

    nonzero = delay_pattern[0]
    for i in np.arange(0, delay_pattern.shape[0]):
        if delay_pattern[i] < 1e-6:
            delay_pattern[i] = nonzero
        else:
            delay_pattern[i] = 0

    # Transforma os atrasos em indices a deslocar as linhas:
    sample_time = data.inspection_params.sample_time

    # Extrai a imagem:
    img = data.ascan_data_sum[:, :, 0]


    # Desloca a imagem para ajustar os tempos de percusso na sapata:
    img = shiftImage(img, delay_pattern, sample_time)

    # Processamento da Imagem:
    gamma = 2
    img = img ** gamma # Gamma da imagem
    new_img = (img - img.min())/(img.max() - img.min()) * 255 # Autocontraste
    new_img = np.log10(new_img + new_img.min() + 1e-5)
    # # Encontra o histograma da imagem:
    # histr = cv2.calcHist([new_img], [0], None, [256], [0, 256])

    # Plota a imagem:
    X = angles
    Y = t_span
    angles = betas
    plt.imshow(new_img, interpolation='none', aspect='auto', extent=[angles.min(), angles.max(), t_span[-1], t_span[0]])
    # plt.pcolormesh(X,Y,new_img)

    #, extent=[angles.min(), angles.max(), t_span[-1], t_span[0]]
    # Máximo por coluna:
    for col_idx in np.arange(0, new_img.shape[1]):
        column = new_img[:, col_idx]
        max_row_idx = np.argmax(column)

        theta_max = col_idx * (angles.max() - angles.min()) / new_img.shape[1] + angles.min()
        time_max = max_row_idx * (t_span[-1] - t_span[0]) / new_img.shape[0] + 40
        plt.scatter(theta_max, time_max, color=[1, 0, 0], s=1.5)


    # Plota os limites da incidência direta/indireta:
    y = t_span
    theta_lim = np.abs(angles[9])
    x1 = -theta_lim * np.ones_like(y)
    x2 = theta_lim * np.ones_like(y)
    note_color = [1, 0, 1]
    plt.plot(x1, y, ':', x2, y, ':', color=note_color)


    plt.grid()
    plt.xlabel("Ângulo de Varredura da Tubulação em graus")
    plt.ylabel("Tempo de ensaio em $\mu$s")