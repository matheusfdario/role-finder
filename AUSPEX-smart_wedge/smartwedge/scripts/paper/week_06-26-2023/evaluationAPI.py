from framework import file_m2k
import numpy as np
import matplotlib.pyplot as plt
from framework.post_proc import envelope

# Script faz a análise usando API do ensaio deslocando o furo para margem da imagem.

def crop_ascan(ascan, t_span, t0=None, tf=None):
    if t0 is not None and tf is not None:
        t0_idx = np.argmin(np.power(t_span - t0, 2))
        tf_idx = np.argmin(np.power(t_span - tf, 2))
        return ascan[t0_idx:tf_idx, :]


def plot_echoes(t_base, t_echoes, n_echoes=3, color='blue', label='_', xbeg=-40, xend=40, alpha=.3):
    x = np.arange(xbeg, xend, 1e-1)
    for n in range(n_echoes):
        y = np.ones_like(x) * (t_base + t_echoes * (n + 1))
        plt.plot(x, y,  ':', color=color, label=label, alpha=alpha)
    if label != "_":
        plt.legend()

def api_func(img, corners, thresh=.5, drawSquare=True):
    north_east_corner = corners[0]
    south_west_corner = corners[1]
    img_cropped = img[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]]
    local_max = np.max(img_cropped)
    maxLocationCoord = np.where(img_cropped==local_max)
    maxLocation = maxLocationCoord[1] + north_east_corner[1]
    img_cropped_masked = img_cropped > thresh * local_max
    img_masked = np.zeros_like(img)
    img_masked[north_east_corner[0]:south_west_corner[0], north_east_corner[1]:south_west_corner[1]] += img_cropped_masked
    api = np.sum(img_masked * 1.0) / len(img_masked)

    if drawSquare:
        width = 1
        scale_factor = int(img.shape[0]/img.shape[1])
        img_masked[north_east_corner[0] - width * scale_factor : south_west_corner[0] + width * scale_factor,
                   north_east_corner[1] - width : north_east_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor : south_west_corner[0] + width * scale_factor,
                   south_west_corner[1] - width: south_west_corner[1] + width] = 1

        img_masked[north_east_corner[0] - width * scale_factor : north_east_corner[0] + width * scale_factor,
                   north_east_corner[1] - width : south_west_corner[1] + width] = 1

        img_masked[south_west_corner[0] - width * scale_factor : south_west_corner[0] + width * scale_factor,
                   north_east_corner[1] - width: south_west_corner[1] + width] = 1

    return api, maxLocation, img_masked, local_max
# Análise dos ascans:

experiment_root = "/media/tekalid/Data/smartwedge_data/06-26-2023/"
# experiment_ref_1 = "ref_onda_com_foco_submerso.m2k"
# experiment_1 = "onda_com_foco_submerso.m2k"
# experiment_ref = "ref_onda_plana_submerso.m2k"
experiment_1 = "posicao_01.m2k"
experiment_2 = "posicao_02.m2k"
experiment_3 = "posicao_03.m2k"
experiment_4 = "posicao_04.m2k"
experiment_5 = "posicao_05.m2k"
experiment_6 = "posicao_06.m2k"
experiment_ref = "referencia.m2k"
betas = np.linspace(-40, 40, 161)



data_experiment_2 = file_m2k.read(experiment_root + experiment_2, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

data_experiment_3 = file_m2k.read(experiment_root + experiment_3, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

data_experiment_4 = file_m2k.read(experiment_root + experiment_4, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

data_experiment_5 = file_m2k.read(experiment_root + experiment_5, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

data_experiment_6 = file_m2k.read(experiment_root + experiment_6, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

data_ref = file_m2k.read(experiment_root + experiment_ref, type_insp='contact', water_path=0, freq_transd=5,
                                  bw_transd=0.5, tp_transd='gaussian', sel_shots=0)


t_span_original = data_ref.time_grid

# Corta o scan e timegrid para range desejado:
t0 = 50
tend = 50 + 15

# Corta o A-scan para limites definidos:
data_ref.ascan_data = crop_ascan(data_ref.ascan_data, t_span_original, t0, tend)


# New t_span
t_span = crop_ascan(t_span_original, t_span_original, t0, tend)

# Operações
log_cte = .5

# Definir mesma colorbar:
vmin_sscan = 0
vmax_sscan = 5.5

# Cantos das caixas que irão conter as falhas:
corners = [
    [(345, 60), (480, 95)],  # [(Row, column), (Row, column)] and [(North-east), (South-west)]
    [(345, 70), (480, 120)],  #
    [(345, 80), (480, 120)],
    [(345, 100), (480, 140)],
    [(345, 110), (480, 160)],
    [(345, 110), (480, 160)]
]

api_vec = np.zeros(16)
maxAng = np.zeros_like(api_vec)
maxPixel = np.zeros_like(api_vec)
m = 0

plt.figure()
for i in range(0, 16):
    print(f"Progresso:{i/16 * 100:.1f} %")
    j = i + 1
    experiment_name = f"posicao_{j:02d}.m2k"
    data_experiment = file_m2k.read(experiment_root + experiment_name, type_insp='contact', water_path=0, freq_transd=5,
                                      bw_transd=0.5, tp_transd='gaussian', sel_shots=0)

    # Corta o A-scan para limites definidos:
    data_experiment.ascan_data = crop_ascan(data_experiment.ascan_data, t_span_original, t0, tend)

    # Faz a operação de somatório + envoltória:
    sscan_exp = envelope(np.sum(data_experiment.ascan_data - data_ref.ascan_data, axis=2), axis=0)
    sscan_exp_log = np.log10(sscan_exp + log_cte)

    # APlicação da API:
    corners = [(300, 60 + i * 5), (430, 95 + i * 5)]
    api_vec[i], maxAngIdx, img_masked, maxPixel[i] = api_func(sscan_exp, corners, thresh=.5)
    maxAng[i] = betas[maxAngIdx]

    if j % 2 == 0:
        m += 1
        if m == 5:
            pass
        print(m)
        plt.subplot(2, 8, m)
        plt.title(f"S-scan da posição {j}")
        plt.imshow(sscan_exp_log, extent=[-40, 40, t_span[-1][0], t_span[0][0]], cmap='magma', aspect='auto',
                   interpolation="None", vmin=vmin_sscan, vmax=vmax_sscan)
        plot_echoes(60, 0, n_echoes=1, color='blue', xbeg=-40, xend=40)

        if j == 1:
            plt.ylabel(r"Tempo em $\mu s$")
            plt.xlabel(r"Ângulo de varredura da tubulação")

        plt.subplot(2, 8, m + 8)
        plt.title(f"API={api_vec[i]:.4f}")
        plt.imshow(img_masked, extent=[-40, 40, t_span[-1][0], t_span[0][0]], aspect='auto', interpolation="None")

        if i == 1:
            plt.ylabel(r"Tempo em $\mu s$")
            plt.xlabel(r"Ângulo de varredura da tubulação")

plt.figure()
plt.title("API em função do ângulo de varredura.")
plt.plot(maxAng, api_vec, 'o:')
plt.xticks(maxAng)
plt.ylabel("API")
plt.xlabel(r"Ângulo de varredura da tubulação")
plt.grid()

plt.figure()
plt.title("Valor do máximo de intensidade do pixel (não está em escala log).")
plt.plot(maxAng, maxPixel, 'o:', color='r')
plt.xticks(maxAng)
plt.ylabel("Intensidade do pixel")
plt.xlabel(r"Ângulo de varredura da tubulação")
plt.grid()


