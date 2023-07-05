import numpy as np
import matplotlib.pyplot as plt



def contact_point_focus_focal_law(focus_distances, focus_angles_deg, velocity, elem_pos):
    # Gera lei focal para onda plana no caso de ensaio com contato (possui atrasos negativos):
    r1max = 0
    for focus_distance, focus_angle_deg in zip(focus_distances, focus_angles_deg):
        focus_angle = np.deg2rad(90 - np.abs(focus_angle_deg))

        r = focus_distance
        theta = focus_angle

        focus_coord = np.array([r * np.cos(theta), r * np.sin(theta)])  # Posição do foco em coordenadas cartesianas;
        r1 = np.max(np.array([np.linalg.norm(X - focus_coord) for X in elem_pos]))
        if r1 >= r1max:
            r1max = r1



    focal_law = np.zeros((focus_angles_deg.shape[0], elem_pos.shape[0]))
    idx = 0
    for focus_distance, focus_angle_deg in zip(focus_distances, focus_angles_deg):
        focus_angle = np.deg2rad(90 - np.abs(focus_angle_deg))

        r = focus_distance
        theta = focus_angle

        focus_coord = np.array([r * np.cos(theta), r * np.sin(theta)]) # Posição do foco em coordenadas cartesianas;

        r1 = r1max
        ri = np.array([np.linalg.norm(X - focus_coord) for X in elem_pos])
        current_focal_law = (r1 - ri)/velocity

        # focal_law = np.abs(np.max(focal_law) - focal_law) # A lei focal deverá ter uma relação inversa ao tempo de percusso
        # entre o elemento do transdutor e o foco
        current_focal_law = current_focal_law / 1e-6  # Converte para us

        if focus_angle_deg < 0:
            current_focal_law = current_focal_law[::-1]

        focal_law[idx, :] = current_focal_law
        idx += 1

    return focal_law



def contact_planewave_focal_law(planewave_angle_deg, velocity, elem_pos):
    try:
        if not type(planewave_angle_deg) == np.array:
            planewave_angle = np.array(planewave_angle_deg, dtype=float)
    except ValueError:
        raise TypeError("Não foi possível converter o argumento ``planewave_angles`` para ``numpy.array``.")

    # Gera lei focal para onda plana no caso de ensaio com contato (possui atrasos negativos):
    focal_law = np.array([np.sin(np.deg2rad(angle)) * elem_pos / velocity for angle in planewave_angle])
    focal_law = focal_law + np.abs(np.min(focal_law))  # Converte para uma lei focal com tempos positivos
    focal_law = focal_law / 1e-6  # Converte para us
    return focal_law


def compensate_time(ang_focus_deg, smartwedge):
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
        time_bw = np.linalg.norm(intersect - smartwedge.B)/(smartwedge.wedge_cl * 1e3) * 1e-3 # Tempo entre centro do transdutor e incidência direta
        time_ws = (np.linalg.norm(smartwedge.A - intersect) - smartwedge.r0)/(smartwedge.coupling_cl * 1e3) * 1e-3 # Tempo entre tubulação e sapata
        compesation_time = time_bw + time_ws

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        time_bd = np.linalg.norm(intersect - smartwedge.B)/(smartwedge.wedge_cl * 1e3) * 1e-3 # Distância entre centro do transdutor e elipse
        time_dw = (np.linalg.norm(smartwedge.A - intersect) - radius)/(smartwedge.wedge_cl * 1e3) * 1e-3 # Dist. entre elipse e borda da sapata
        time_ws = (radius - smartwedge.r0)/(smartwedge.coupling_cl * 1e3) * 1e-3 # Distância entre sapata e tubulação
        compesation_time = time_bd + time_dw + time_ws

    return compesation_time

def compensate_time_wedge(ang_focus_deg, smartwedge):
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
        time_bw = np.linalg.norm(intersect - smartwedge.B)/(smartwedge.wedge_cl * 1e3) * 1e-3 # Tempo entre centro do transdutor e incidência direta

        compesation_time = time_bw

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        time_bd = np.linalg.norm(intersect - smartwedge.B)/(smartwedge.wedge_cl * 1e3) * 1e-3 # Distância entre centro do transdutor e elipse
        time_dw = (np.linalg.norm(smartwedge.A - intersect) - radius)/(smartwedge.wedge_cl * 1e3) * 1e-3 # Dist. entre elipse e borda da sapata
        compesation_time = time_bd + time_dw

    return compesation_time

def compute_focus_dist(raio, ang_focus_deg, smartwedge):
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
        dist_bw = np.linalg.norm(intersect - smartwedge.B) * 1e-3 # Tempo entre centro do transdutor e incidência direta
        dist_ws = (np.linalg.norm(smartwedge.A - intersect) - smartwedge.r0) * 1e-3 # Tempo entre tubulação e sapata
        dist = dist_bw + dist_ws + (smartwedge.r0 - raio)

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        dist_bd = np.linalg.norm(intersect - smartwedge.B) * 1e-3 # Distância entre centro do transdutor e elipse
        dist_dw = (np.linalg.norm(smartwedge.A - intersect) - radius) * 1e-3 # Dist. entre elipse e borda da sapata
        dist_ws = (radius - smartwedge.r0) * 1e-3 # Distância entre sapata e tubulação
        dist = dist_bd + dist_dw + dist_ws + (smartwedge.r0 - raio)

    # Distância em metros
    return dist

def compute_total_dist(ang_focus_deg, smartwedge):
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
        dist_bw = np.linalg.norm(intersect - smartwedge.B) * 1e-3 # Tempo entre centro do transdutor e incidência direta
        dist_ws = (np.linalg.norm(smartwedge.A - intersect) - smartwedge.r0) * 1e-3 # Tempo entre tubulação e sapata
        dist = dist_bw + dist_ws + smartwedge.r0 * 1e-3

    # Checa se o foco está na direção da incidÊncia indireta:
    elif angle_b <= ang_focus_deg <= angle_c:
        focus_angle, intersect = smartwedge.compute_entrypoints(ang_focus_deg)
        dist_bd = np.linalg.norm(intersect - smartwedge.B) * 1e-3 # Distância entre centro do transdutor e elipse
        dist_dw = (np.linalg.norm(smartwedge.A - intersect) - radius) * 1e-3 # Dist. entre elipse e borda da sapata
        dist_ws = (radius - smartwedge.r0) * 1e-3 # Distância entre sapata e tubulação
        dist = dist_bd + dist_dw + dist_ws + smartwedge.r0 * 1e-3

    # Distância em metros
    return dist

def generate_law(filename, focal_law, elem_range=None):
    header = [
        "# LOIS DE RETARD \n",
        "Version 1.0 \n",
        "numR\t"
        "numS\t"
        "numT\t"
        "numL\t"
        "numV\t"
        "retE\t"
        "ampE\t"
        "retR\t"
        "ampR\n"
    ]

    if elem_range is None:
        elem_range = [0, focal_law.shape[1] - 1]

    with open(filename + ".law", "w") as file:
        file.writelines(header)

        for shot in range(0, focal_law.shape[0]):
            for elem_idx in range(elem_range[0], elem_range[-1]+1):
                numR = 0 #
                numS = 0 #
                numT = shot # Shot
                numL = 0 #
                numV = elem_idx + 1 # Índice do Emissor
                retE = focal_law[shot, elem_idx] # Lei focal na Emissão
                ampE = 1 # Ganho na Emissão
                retR = focal_law[shot, elem_idx] # Lei focal na Recepção
                ampR = 1 # Ganho na Recepção
                datum = [numR, numS, numT, numL, numV, retE, ampE, retR, ampR]
                data_line = [f"{datum[i]}" + "\t" for i in range(0, len(datum) - 1)]
                data_line.append(f"{datum[-1]}\n")
                file.writelines(data_line)


def shiftImage(sectorial_img, delays, time_sample):
    shifts = np.round(delays / time_sample).astype(int)
    m, n = sectorial_img.shape
    m_new = np.round(m + np.max(shifts)).astype(int)
    shifted_img = np.zeros((m_new, n))

    for i in np.arange(0, n):
        shifted_img[shifts[i]:shifts[i] + m, i] = sectorial_img[:, i]
    shifted_img = shifted_img[:m, :n]
    return shifted_img

