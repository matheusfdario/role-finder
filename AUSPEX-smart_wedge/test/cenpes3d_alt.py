import numpy as np
import time
from framework import file_m2k, post_proc, pre_proc
from framework.data_types import ImagingROI
from imaging import tfm
from imaging import tfmcuda
from surface.surface import SurfaceType
from surface.surface import Surface
from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line
import open3d as o3d
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt

lista = ['/home/hector/PycharmProjects/AUSPEX/data/cenpes/CP2_75_Water_90.m2k',
         '/home/hector/PycharmProjects/AUSPEX/data/cenpes/CP2_75_Water_90_shifted_5_mm.m2k']
surftop = []
surfbot = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []
surfy = [0, 5]
step = 1
stepx = 0.1
stepy = 0.1
stepz = 0.1
l = 0

width = 28.0
height = 10.0

for k, file in enumerate(lista):
    print(file)
    data = file_m2k.read(file,
                         freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
    _ = pre_proc.hilbert_transforms(data)
    corner_roi = np.array([-14.0, 0.0, 15.0])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height/stepz), w_len=int(width/stepx),
                     depth=1.0, d_len=1)
    surfx = roi.w_points
    data.inspection_params.type_insp = 'immersion'
    data.inspection_params.coupling_cl = 1483.0
    data.surf = Surface(data, surf_type=SurfaceType.CIRCLE_NEWTON)
    data.surf.fit()

    t0 = time.time()
    chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0, c=5885.5)
    print(time.time()-t0)

    yt = post_proc.envelope(data.imaging_results[chave].image)
    if yt.sum()>0:
        y = yt/yt.max()

    # Surface Estimation
    a = img_line(y)
    z = roi.h_points[a[0].astype(int)]
    w = a[1]
    lamb = 10
    rho = 100
    print(f'Estimating Surface')
    if k>0:
        bot_ant = bot
        top_ant = top
    # t0 = time.time()
    bot, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w.ravel(), z, lamb, x0=z, rho=rho, eta=.999,
                                                             itmax=250, tol=1e-9, max_iter_cg=500)
    # print(time.time()-t0)
    top = np.interp(surfx, data.surf.x_discr, data.surf.z_discr)
    print(f'Making points')
    surftop.extend([(surfx[i], top[i], surfy[k]) for i in range(len(surfx))])
    surfbot.extend([(surfx[i], bot[i], surfy[k]) for i in range(len(surfx))])
    L = int(5/stepy)
    if k > 0:
        for l in range(L - 1):
            auxtop = (l + 1) / L * (top - top_ant) + top_ant
            auxbot = (l + 1) / L * (bot - bot_ant) + bot_ant
            auxy = (l + 1) / L * (surfy[k] - surfy[k - 1]) + surfy[k - 1]
            surftop.extend([(surfx[i], auxtop[i], auxy) for i in range(len(surfx))])
            surfbot.extend([(surfx[i], auxbot[i], auxy) for i in range(len(surfx))])
            psid1.extend([(surfx[0], i, auxy) for i in np.arange(auxtop[0]+stepz, auxbot[0]-stepz, stepz)])
            psid2.extend([(surfx[-1], i, auxy) for i in np.arange(auxtop[-1]+stepz, auxbot[-1]-stepz, stepz)])
    for j, wp in enumerate(surfx):
        if k == 0:
            pfac1.extend([(surfx[j], i, surfy[k]) for i in np.arange(top[j]+stepz, bot[j]-stepz, stepz)])
        if k == len(lista)-1:
            pfac2.extend([(surfx[j], i, surfy[k]) for i in np.arange(top[j]+stepz, bot[j]-stepz, stepz)])

pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]
steps = (stepx, stepy, stepz)
# Gerar normais e mesh
pcd = pl2pc(pts, steps)
o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
mesh = p2m(pcd, depth=8)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
# o3d.io.write_triangle_mesh('/home/hector/Documents/point_cloud/cenpes.stl', mesh, print_progress=True)
