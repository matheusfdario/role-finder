import numpy as np
from .smartwedge import Smartwedge

def _create_zerolike_img(raw_data, type=float):
    if len(raw_data.shape) == 4:
        zero_img = np.zeros_like(raw_data[:, :, :, 0], dtype=type)
    elif len(raw_data.shape) == 3:
        zero_img = np.zeros_like(raw_data[:, :, 0], dtype=type)
    return zero_img


def transd2tube_timeofflight(ang_focus_deg, smartwedge):
    # Encontra o tempo em que o som percorre (ida e volta) entre o centro do transdutor e o centro da tubulação
    # para um dado ângulo de disparo em relaçao ao centro da tubulação.
    # r_focus é a distância do foco para o centro da tubulação
    # ang_focus_deg é posição angular da leitura em relação ao centro da tubulação
    ang_focus_deg = np.abs(ang_focus_deg)

    angle_a = 0
    angle_b = np.rad2deg(np.arctan(smartwedge.N[1] / smartwedge.N[0]))
    angle_c = 90

    radius = np.linalg.norm(smartwedge.N)
    # Checa se o foco está na direção da incidÊncia direta:
    if angle_a <= ang_focus_deg < angle_b:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        time_bw = np.linalg.norm(intersect - smartwedge.B) / (
                    smartwedge.wedge_cl * 1e3) * 1e-3  # Tempo entre centro do transdutor e incidência direta
        time_ws = (np.linalg.norm(smartwedge.A - intersect) - smartwedge.r0) / (
                    smartwedge.coupling_cl * 1e3) * 1e-3  # Tempo entre tubulação e sapata
        time_of_flight = time_bw + time_ws

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        time_bd = np.linalg.norm(intersect - smartwedge.B) / (
                    smartwedge.wedge_cl * 1e3) * 1e-3  # Distância entre centro do transdutor e elipse
        time_dw = (np.linalg.norm(smartwedge.A - intersect) - radius) / (
                    smartwedge.wedge_cl * 1e3) * 1e-3  # Dist. entre elipse e borda da sapata
        time_ws = (radius - smartwedge.r0) / (smartwedge.coupling_cl * 1e3) * 1e-3  # Distância entre sapata e tubulação
        time_of_flight = time_bd + time_dw + time_ws

    return time_of_flight


def transd2wedge_timeofflight(ang_focus_deg, smartwedge):
    # Encontra o ângulo e a distância do foco em relação ao referencial do transdutor.
    # r_focus é a distância do foco para o centro da tubulação
    # ang_focus_deg é posição angular da leitura em relação ao centro da tubulação
    ang_focus_deg = np.abs(ang_focus_deg)

    angle_a = 0
    angle_b = np.rad2deg(np.arctan(smartwedge.N[1] / smartwedge.N[0]))
    angle_c = 90

    radius = np.linalg.norm(smartwedge.N)
    # Checa se o foco está na direção da incidÊncia direta:
    if angle_a <= ang_focus_deg < angle_b:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        time_bw = np.linalg.norm(intersect - smartwedge.B) / (
                    smartwedge.wedge_cl * 1e3) * 1e-3  # Tempo entre centro do transdutor e incidência direta

        time_of_flight = time_bw

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        time_bd = np.linalg.norm(intersect - smartwedge.B) / (
                    smartwedge.wedge_cl * 1e3) * 1e-3  # Distância entre centro do transdutor e elipse
        time_dw = (np.linalg.norm(smartwedge.A - intersect) - radius) / (
                    smartwedge.wedge_cl * 1e3) * 1e-3  # Dist. entre elipse e borda da sapata
        time_of_flight = time_bd + time_dw

    return time_of_flight


def shift_image(sectorial_img, delays, sampling_period):
    shifts = np.round(delays / sampling_period).astype(int)
    m, n = sectorial_img.shape
    m_new = np.round(m + np.max(shifts)).astype(int)
    shifted_img = np.zeros((m_new, n))

    for i in np.arange(0, n):
        shifted_img[shifts[i]:shifts[i] + m, i] = sectorial_img[:, i]
    shifted_img = shifted_img[:m, :n] # Corta a imagem
    return shifted_img


def compute_shift_law(sm, tube_angles_deg):
    roll_partern = np.zeros_like(tube_angles_deg, dtype=float)
    time_spent = np.zeros_like(roll_partern)  # tempo em que o som está percorrendo o interior da sapata
    wedge_time = np.zeros_like(roll_partern)
    for i, bet in enumerate(tube_angles_deg):
        roll_partern[i] = transd2tube_timeofflight(bet, sm) * 2
        time_spent[i] = roll_partern[
                            i] / 1e-6  # Tempo que o som demorou para percorrer do centro da tubulação ao centro
        # do transdutor.
        wedge_time[i] = transd2wedge_timeofflight(bet, sm) / 1e-6 * 2  # Tempo que o som demorou para percorrer do
        # centro do transdutor até a borda da sapata.

    roll_partern = roll_partern - roll_partern.min()
    roll_partern = roll_partern / 1e-6

    nonzero = roll_partern[0]
    for i in np.arange(0, roll_partern.shape[0]):
        if roll_partern[i] < 1e-6:
            roll_partern[i] = nonzero
        else:
            roll_partern[i] = 0
    return roll_partern, wedge_time, time_spent


def generate_mean_img_from_npz(npz_data, datum_name, sum_data=False):
    for i in range(len(datum_name)):
        print(f"Lendo {datum_name[i]}")
        data_name = datum_name[i]
        img = _create_zerolike_img(npz_data[data_name])
        n_shots = npz_data[data_name].shape[-1]

        if sum_data and len(img.shape) == 4:
            img += np.sum(npz_data[data_name][:, :, :, :], axis=3) / n_shots
        elif not sum_data and len(img.shape) == 4:
            img += np.sum(np.sum(npz_data[data_name][:, :, :, :], axis=3), axis=2) / n_shots
        elif len(img.shape) == 3:
            img += np.sum(npz_data[data_name][:, :, :, :], axis=3) / n_shots
        else:
            raise ValueError("Dimensão de dados incopatível com parâmetros de chamada da função.")

    return img / (i + 1)


def generate_piecewise_linear_radii_span(sm, t_span, transd2wedge_tof, transd2tube_tof,
                                         tube_cl, tube_wall_width):
    d = np.zeros_like(t_span)
    for i, t in enumerate(t_span):
        if t_span[0] <= t <= transd2wedge_tof:
            d[i] = t * sm.wedge_cl  # em mm
            # print(d[i])
        elif transd2wedge_tof < t <= transd2tube_tof:
            t_prime = t - transd2wedge_tof
            d[i] = t_prime * sm.coupling_cl + transd2wedge_tof * sm.wedge_cl  # em mm
        elif transd2tube_tof < t <= transd2tube_tof + (2 * tube_wall_width) / tube_cl:
            t_prime = t - transd2tube_tof
            d[i] = (t_prime * tube_cl + \
                    (transd2wedge_tof * sm.wedge_cl +
                     (transd2tube_tof - transd2wedge_tof) * sm.coupling_cl
                     ))
        elif transd2tube_tof + (2 * tube_wall_width) / tube_cl < t <= t_span[-1]:
            t_prime = t - transd2tube_tof + (2 * tube_wall_width) / tube_cl
            d[i] = (t_prime * sm.coupling_cl + \
                    (transd2wedge_tof * sm.wedge_cl +
                     (transd2tube_tof - transd2wedge_tof) * sm.coupling_cl +
                     tube_wall_width
                     ))


    beg_time = transd2wedge_tof + 1
    beg_idx = np.argmin(np.power(t_span - beg_time, 2))
    radii = d[beg_idx:]
    total_dist = sm.compute_total_dist(60) * 2e3
    tube_radii = (total_dist - radii) / 2
    return tube_radii, beg_idx


def combine_sm_images(dir_img, ind_img, sm: Smartwedge, tube_ang_deg: float):
    img = np.zeros_like(dir_img)
    col2change = sm.is_indirect_inc(tube_ang_deg)
    for j in range(img.shape[1]):
        if col2change[j] == True:
            img[:, j] = ind_img[:, j]
        else:
            img[:, j] = dir_img[:, j]
    return img