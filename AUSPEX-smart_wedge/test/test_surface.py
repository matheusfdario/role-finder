import numpy as np
from matplotlib import pyplot as plt
import time
from framework import file_m2k, file_civa, post_proc
from framework.data_types import ImagingROI
from imaging import tfm, cumulative_tfm
from surface.surface import SurfaceType
from surface.surface import Surface
from framework.pre_proc import hilbert_transforms
def angleVec(data):
    k = np.array([0, 0, 1])
    v = np.array([data.surf.surfaceparam.a, data.surf.surfaceparam.b, 1])
    inner_product = np.dot(k, v)
    den = np.linalg.norm(k)*np.linalg.norm(v)
    theta_rad = np.arccos(inner_product/den)
    return np.rad2deg(theta_rad)

# data = file_civa.read('/home/hector/PycharmProjects/AUSPEX/data/Simulation_DadosCENPES_imersion.civa')
# data = file_m2k.read('/home/hector/PycharmProjects/AUSPEX/data/tubo_acrilico_com_dentes_FMC.m2k',
#                      freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
# data.ascan_data = hilbert_transforms(data, [0])
shots = range(0, 25)
print('Loading File')
data = file_civa.read('/home/hector/PycharmProjects/AUSPEX/data/pecas_acrilico_old/peca11.civa', sel_shots=shots)
shots = np.asarray(shots) - shots[0]
print('Applying Hilbert Transform')
corner_roi = np.array([-30.0, 0.0, 80.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=10.0, width=60.0, h_len=200, w_len=600, depth=10.0, d_len=1)
data.surf = Surface(data, surf_type=SurfaceType.ATFM)
data.surf.fit(roi=roi, shot=shots)

def test2D():
    data = file_civa.read(f"C:/virtualmachine/SharedFolder/cilindro.civa")
    data.surf = Surface(data, surf_type=SurfaceType.LINE_NEWTON)
    data.surf.fit()

data = test3D()
theta = angleVec(data)
print(f"theta = {theta}, a = {data.surf.surfaceparam.a}, b={data.surf.surfaceparam.b} c = {data.surf.surfaceparam.c}")