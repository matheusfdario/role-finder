# Graus usados 101,30
# 25 passos de 5mm com raio de aprox. 70,7mm

import numpy as np
import time
from framework import file_m2k, post_proc, pre_proc
from framework.data_types import ImagingROI
from imaging import tfm
from surface.surface import SurfaceType
from surface.surface import Surface
from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line
import open3d as o3d
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt

from scipy.signal import find_peaks

def img_line_Malkin(image):
    ind = np.zeros((2, image.shape[1]))
    x = np.arange(image.shape[1], dtype=np.int16)
    ind[0] = np.arange(image.shape[1])
    h = np.median(np.log10(image))/2
    image0 = image*(np.log10(image)>h)
    for i in x:
        aux = find_peaks(image0[:, i], height=h)
        if len(aux[0])>0:
            ind[1, i] = aux[0][0]
        else:
            ind[1, i] = np.nan
    ind = np.delete(ind, x[np.isnan((ind[1,:]))], 1)
    z = np.interp(x, ind[0], ind[1])
    return z

lista = np.linspace(0, 101.30, 25)
radius = 70.5
waterpath = 10
surfy = np.sin(np.array(lista)*np.pi/180)
step = 1
stepx = 0.1
stepy = 0.1
stepz = 0.1
l = 0

width = 30.0
file = "/home/hector/PycharmProjects/AUSPEX/data/setup_correia_30dB_middensity.m2k"
data = file_m2k.read(file, freq_transd=5, bw_transd=0.7, tp_transd='gaussian')
_ = pre_proc.hilbert_transforms(data, shots=np.arange(data.inspection_params.step_points.shape[0]))
data.inspection_params.type_insp = 'immersion'
data.inspection_params.coupling_cl = 1483.0
data.surf = Surface(data, surf_type=SurfaceType.LINE_OLS)

surftop = []
surfbot = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []
for k, angle in enumerate(lista):
    print(f'Angle: {angle:.2f} - {k}/{len(lista)} - {k/len(lista)*100:.2f}%', end='\r', flush=True)
    data.surf.fit(shot=k)
    waterpath = data.surf.surfaceparam.b
    height = 22.05 + waterpath - 25 #11.2
    corner_roi = np.array([-15.0, 0.0, 25])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height / stepz), w_len=int(width / stepx),
                     depth=1.0, d_len=1)
    surfx = roi.w_points

    chave = tfm.tfm_kernel(data, roi=roi, output_key=k, sel_shot=k, c=5900)
    yt = post_proc.envelope(data.imaging_results[chave].image)
    y = yt/yt.max()
    # # Surface Estimation
    a = img_line(y)
    z = roi.h_points[a[0].astype(int)]
    zm = roi.h_points[img_line_Malkin(y).astype(int)]
    w = a[1]
    lamb = 0.5
    rho = 100
    # print(f'\tEstimating Surface')
    if k>0:
        bot_ant = bot
        top_ant = top
    bot, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w.ravel(), z, lamb, x0=z, rho=rho, eta=.999,
                                                             itmax=250, tol=1e-9, max_iter_cg=500)
    top = np.interp(surfx, data.surf.x_discr, data.surf.z_discr)
    # print(f'\tMaking points')
    surftop.extend([(surfx[i], (radius+waterpath-top[i])*np.cos(angle*np.pi/180), (radius+waterpath-top[i])*np.sin(angle*np.pi/180))
                    for i in range(len(surfx))])
    surfbot.extend([(surfx[i], (radius+waterpath-bot[i])*np.cos(angle*np.pi/180), (radius+waterpath-bot[i])*surfy[k]) for i in range(len(surfx))])
    psid1.extend([(surfx[0], (radius + waterpath - i) * np.cos(angle * np.pi / 180),
                   (radius + waterpath - i) * np.sin(angle * np.pi / 180))
                  for i in np.arange(top[0] + stepz, bot[0] - stepz, stepz)])
    psid2.extend([(surfx[-1], (radius + waterpath - i) * np.cos(angle * np.pi / 180),
                   (radius + waterpath - i) * np.sin(angle * np.pi / 180))
                  for i in np.arange(top[-1] + stepz, bot[-1] - stepz, stepz)])
    for j, wp in enumerate(surfx):
        if k == len(lista) - 1:
            pfac2.extend([(surfx[j], (radius + waterpath - i) * np.cos(angle * np.pi / 180),
                           (radius + waterpath - i) * np.sin(angle * np.pi / 180))
                          for i in np.arange(top[-1] + stepz, bot[-1] - stepz, stepz)])
        if k == 0:
            pfac1.extend([(surfx[j], (radius + waterpath - i) * np.cos(angle * np.pi / 180),
                           (radius + waterpath - i) * np.sin(angle * np.pi / 180))
                          for i in np.arange(top[0] + stepz, bot[0] - stepz, stepz)])
    L = 3
    if k > 0 and k < len(lista) - 1:
        for l in range(L):
            auxangle = (l+1)/L * (lista[k] - lista[k-1]) + lista[k]
            auxtop = (l + 1) / L * (top - top_ant) + top_ant
            auxbot = (l + 1) / L * (bot - bot_ant) + bot_ant
            auxy = (l + 1) / L * (surfy[k] - surfy[k - 1]) + surfy[k - 1]
            surftop.extend([(surfx[i], (radius + waterpath - auxtop[i]) * np.cos(auxangle * np.pi / 180),
                             (radius + waterpath - auxtop[i]) * np.sin(auxangle * np.pi / 180)) for i in range(len(surfx))])
            surfbot.extend([(surfx[i], (radius + waterpath - auxbot[i]) * np.cos(auxangle * np.pi / 180),
                             (radius + waterpath - auxbot[i]) * np.sin(auxangle * np.pi / 180)) for i in range(len(surfx))])
            psid1.extend([(surfx[0], (radius+waterpath-i)*np.cos(auxangle*np.pi/180), (radius+waterpath-i)*np.sin(auxangle*np.pi/180))
                          for i in np.arange(auxtop[0]+stepz, auxbot[0]-stepz, stepz)])
            psid2.extend([(surfx[-1], (radius+waterpath-i)*np.cos(auxangle*np.pi/180), (radius+waterpath-i)*np.sin(auxangle*np.pi/180))
                          for i in np.arange(auxtop[-1]+stepz, auxbot[-1]-stepz, stepz)])

print(f'Forming Point Cloud with normals')
pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]
steps = (stepx, stepy, stepz)
# Gerar normais e mesh
pcd = pl2pc(pts, steps, orient_tangent=False, xlen=len(surfx), radius_bot=10)
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
mesh = p2m(pcd, depth=8, smooth=True)
mesh = mesh.simplify_quadric_decimation(100000)
triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)
triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
mesh.remove_triangles_by_mask(triangles_to_remove)
mesh.compute_triangle_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)
# o3d.io.write_triangle_mesh('/home/hector/Documents/point_cloud/meiacana_maxdensity_10dB.stl', mesh, print_progress=True)
