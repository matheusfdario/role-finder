from matplotlib.animation import FFMpegWriter
import numpy as np
import matplotlib.pyplot as plt
from framework.post_proc import envelope
from framework import file_m2k
from smartwedge_utils import *
from class_smartwedge import *


def plot_frame(shot, dir_data, ind_data):
    # Imagem da incidência direta:
    dir_sum = dir_data[:, :, :, shot].sum(axis=2)
    dir_shifted = shift_image(dir_sum, shift_pattern, t_span[1]-t_span[0])
    dir_env = envelope(dir_shifted, axis=0)
    dir_log = np.log10(dir_env + 1)



    # Imagem da referência indireta:
    ind_sum = ind_data.sum(axis=2)
    ind_shifted = shift_image(ind_sum, shift_pattern, t_span)
    ind_env = envelope(ind_shifted, axis=0)
    ind_log = np.log10(ind_env + 1)

    # Gera o domínio linear por partes da imagem:
    tube_radii, beg_idx = generate_piecewise_linear_radii_span(sm, t_span, ind_transd2wedge_tof[0],
                                                               ind_trasnd2tube_tof[0], v_steel, tube_wall_width)

    # Plota os resultados:
    plt.pcolormesh(tube_angles_deg, tube_radii, dir_log[beg_idx:, :], cmap="gray")
    plt.title(f"Ensaio = {file_name} /n shot = {shot}")
    plt.ylabel("distância do centro da tubulação /[mm]")
    plt.xlabel("ângulo de varredura da tubulação /[°]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return None


data_root_m2k = "G:/.shortcut-targets-by-id/1T3yHfBX35wLz0UjXeyUCLflWgR_wB0yv/AUSPEX_OFICIAL/Dados/CENPES/2022 Junho/CENPES 2022-06-29/"
file_name_m2k = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1.m2k"
file_path_m2k = data_root_m2k + file_name_m2k
print("Lendo arquivo .m2k ...")
data_m2k = file_m2k.read(file_path_m2k, read_ascan=False, sel_shots=1, type_insp='contact', water_path=0,
                         freq_transd=5, bw_transd=0.5, tp_transd='gaussian')

# Lê os dados no formato convertido:
data_root = "C:/Users/Kalid/repos/AUSPEX/test/smartwedge_old/"
file_name = "Smart Wedge 29-06-22 100 V Aquisicao 0h v1.npz"
file_path = data_root + file_name

# Lê os dados convertidos do ensaio:
data = np.load(file_path, allow_pickle=True)
direct_datum_name = data.files[2::2]  # Escolhe apenas os dados do tipo multisalvo 0
indirect_datum_name = data.files[1::2][1:]  # Escolhe apenas os dados do tipo multisalvo 1

# Cria objetos para geração de vídeo:
print("Criando objeto para geração de vídeo...")
video_title = "VideoTeste"
metadata = dict(title=video_title, artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=2, metadata=metadata)
fig = plt.figure(figsize=(14, 9))

# Imagens de referência (média):
# print("Gerando arquivos de referência:")
# reference_direct = generate_mean_img_from_npz(data, direct_datum_name, sum_data=False).sum(axis=2)
# reference_indirect = generate_mean_img_from_npz(data, indirect_datum_name, sum_data=False).sum(axis=2)

# Período de amostragem do sinal em micro segundo:
sample_time = data_m2k[0].inspection_params.sample_time

# Tempo de ensaio em micro segundos:
t_span = np.linspace(start=data_m2k[0].inspection_params.gate_start,
                     stop=data_m2k[0].inspection_params.gate_end,
                     num=data_m2k[0].inspection_params.gate_samples)

# Cria um objeto do tipo smartwedge_old:
# Ellipse parameters:
c = 84.28
r0 = 67.15
wc = 6.2
offset = 2
Tprime = 19.5

# Velocidade do som nos materiais em mm/us
v1 = 6.37
v2 = 1.43
v_steel = 5.9

# Espessura da parede da tubulação em milímetros:
tube_wall_width = 16

# Criação do objeto smartwedge_old:
sm = smartwedge(c, r0, wc, v1, v2, Tprime, offset)

# Define ângulos de varredura no referencial da tubulação:
beg_angle = -90
end_angle = 90
step = 5
tube_angles_deg = np.arange(beg_angle, end_angle + step, step)

shift_pattern, ind_transd2wedge_tof, ind_trasnd2tube_tof = compute_shift_law(sm, tube_angles_deg)

# Parâmetros da geração de vídeo:
generate_video = True
shots_per_chunck = 25  # shots por aglomerado
shot = 0  # Para geração de apenas um shot

print("Entrando no loop de geração de vídeo...")
# if generate_video == True:
#     with writer.saving(fig, video_title + ".mp4", dpi=300):
#         for dir_data_name, ind_data_name in zip(direct_datum_name, indirect_datum_name):
#             for shot in range(shots_per_chunck):
#                 dir_data = data[dir_data_name]
#                 ind_data = data[ind_data_name]
#                 plot_frame(shot, dir_data, ind_data)
#                 print(f"frame = {shot + 1}")
#                 writer.grab_frame()
#                 plt.clf()
# else:
#     plot_frame(shot)

dir_data_name = direct_datum_name[0]
ind_data_name = indirect_datum_name[0]