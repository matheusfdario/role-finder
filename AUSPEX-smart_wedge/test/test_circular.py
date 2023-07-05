import numpy as np
from matplotlib import pyplot as plt
import time
from framework import file_m2k, file_civa, post_proc

data = file_civa.read('C:/data/esferas_3D_cir30_FMC.civa')
centers = data.probe_params.elem_center
plt.scatter(centers[:, 0], centers[:, 1])