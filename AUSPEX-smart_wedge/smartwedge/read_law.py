import numpy as np
from .generate_law import *

def read_law(filename, n_elem, n_shot, offset = 2):
    n_data = 9
    focal_law = np.zeros((n_shot * n_elem, n_data))

    header_offset = 3

    with open(filename + ".law", "r") as file:
        lines = file.readlines()
        for i in np.arange(header_offset, n_elem * n_shot + header_offset):
            current_line = lines[i]
            current_line_data = current_line.split("\n")[0].split("\t")
            current_line_data = [float(data) for data in current_line_data]
            numR = current_line_data[0]  #
            numS = current_line_data[1]  #
            numT = current_line_data[2]  # Shot
            numL = current_line_data[3]  #
            numV = current_line_data[4]  # Índice do Emissor
            retE = current_line_data[5]  # Lei focal na Emissão
            ampE = current_line_data[6]  # Ganho na Emissão
            retR = current_line_data[7]  # Lei focal na Recepção
            ampR = current_line_data[8]  # Ganho na Recepção
            focal_law[i-offset-1, :] = current_line_data

    focal_law2 = np.zeros((n_shot, n_elem))
    for n in np.arange(0, n_shot):
        focal_law2[n, :] = focal_law[n*n_elem:(n+1)*n_elem, 5]
    z = 1
    return focal_law2
