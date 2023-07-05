import numpy as np
import time
from framework import file_m2k, post_proc, pre_proc
from framework.data_types import ImagingROI
from imaging import tfmcuda
from surface.surface import SurfaceType
from surface.surface import Surface
from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line
import open3d as o3d
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt
import pickle

positions = [0, 35, 35+35, 35+35+25]
positions = [0]
angles = np.linspace(0, 360, 443)
surftop = []
surfbot = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []
radius = 70.5
waterpath = 10
surfy = np.sin(np.array(angles)*np.pi/180)
step = 1
stepx = 0.1
stepy = 0.1
stepz = 0.1
l = 0

width = 35.0
height = 20.0

# for pos_idx, pos in enumerate(positions):
#     file = f"/home/hector/PycharmProjects/AUSPEX/data/cenpes/spool_ring{pos_idx+1}.m2k"
#
#     for k, angle in enumerate(angles):
#         print(f'Angle: {angle:.2f} - {k+len(angles)*pos_idx}/{len(angles)*len(positions)} - {(k+len(angles)*pos_idx)/(len(angles)*len(positions))*100:.2f}%',
#               end='\r', flush=True)
#         data = file_m2k.read(file, freq_transd=5.2, bw_transd=0.7, tp_transd='gaussian', sel_shots=k)
#         _ = pre_proc.hilbert_transforms(data)
#         data.inspection_params.type_insp = 'immersion'
#         data.inspection_params.coupling_cl = 1483.0
#         data.surf = Surface(data, surf_type=SurfaceType.LINE_OLS)
#         data.surf.fit()
#         waterpath = data.surf.surfaceparam.b
#         height = 20 + waterpath - 15 #11.2
#         corner_roi = np.array([-17.5, 0.0, 15])[np.newaxis, :]
#         # roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height/stepz), w_len=int(width/stepx),
#         #                  depth=1.0, d_len=1)
#         if pos_idx == len(positions)-1:
#             corner_roi = np.array([-7.5, 0.0, 15])[np.newaxis, :]
#         roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height/stepz), w_len=int(width/stepx),
#                      depth=1.0, d_len=1)
#
#         surfx = roi.w_points + pos
#
#         tfmcuda.tfmcuda_kernel(data, roi=roi, output_key=k, sel_shot=0, c=5900)
#         yt = post_proc.envelope(data.imaging_results[k].image)
#         y = yt/yt.max()
#         # # Surface Estimation
#         a = img_line(y)
#         z = roi.h_points[a[0].astype(int)]
#         w = a[1]
#         lamb = 6
#         rho = 100
#         # print(f'\tEstimating Surface')
#         if k>0:
#             bot_ant = bot
#             top_ant = top
#         bot, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w.ravel(), z, lamb, x0=z, rho=rho, eta=.999,
#                                                                  itmax=250, tol=1e-6, max_iter_cg=1500)
#         top = np.interp(surfx, data.surf.x_discr, data.surf.z_discr)
#         top = waterpath*np.ones_like(top)
#         # print(f'\tMaking points')
#         surftop.extend([(surfx[i], -(radius+waterpath-top[i])*np.cos(angle*np.pi/180),
#                          (radius+waterpath-top[i])*np.sin(angle*np.pi/180)) for i in range(len(surfx))])
#         surfbot.extend([(surfx[i], -(radius+waterpath-bot[i])*np.cos(angle*np.pi/180),
#                          (radius+waterpath-bot[i])*surfy[k]) for i in range(len(surfx))])
#         if pos_idx == 0:
#             psid1.extend([(surfx[0], -(radius + waterpath - i) * np.cos(angle * np.pi / 180),
#                            (radius + waterpath - i) * np.sin(angle * np.pi / 180))
#                           for i in np.arange(top[0] + stepz, bot[0] - stepz, stepz)])
#         if pos_idx == len(positions)-1:
#             psid2.extend([(surfx[-1], -(radius + waterpath - i) * np.cos(angle * np.pi / 180),
#                            (radius + waterpath - i) * np.sin(angle * np.pi / 180))
#                           for i in np.arange(top[-1] + stepz, bot[-1] - stepz, stepz)])
        # L = 2
        # if k > 0:
        #     for l in range(L):
        #         auxangle = (l+1)/L * (angles[k] - angles[k-1]) + angles[k]
        #         auxtop = (l + 1) / L * (top - top_ant) + top_ant
        #         auxbot = (l + 1) / L * (bot - bot_ant) + bot_ant
        #         auxy = (l + 1) / L * (surfy[k] - surfy[k - 1]) + surfy[k - 1]
        #         surftop.extend([(surfx[i], (radius + waterpath - auxtop[i]) * np.cos(auxangle * np.pi / 180),
        #                          (radius + waterpath - auxtop[i]) * np.sin(auxangle * np.pi / 180)) for i in
        #                         range(len(surfx))])
        #         # surfbot.extend([(surfx[i], (radius + waterpath - auxbot[i]) * np.cos(auxangle * np.pi / 180),
        #         #                  (radius + waterpath - auxbot[i]) * np.sin(auxangle * np.pi / 180)) for i in
        #         #                 range(len(surfx))])
        #         psid1.extend([(surfx[0], (radius+waterpath-i)*np.cos(angle*np.pi/180), (radius+waterpath-i)*np.sin(angle*np.pi/180))
        #                       for i in np.arange(auxtop[0]+stepz, auxbot[0]-stepz, stepz)])
        #         psid2.extend([(surfx[-1], (radius+waterpath-i)*np.cos(angle*np.pi/180), (radius+waterpath-i)*np.sin(angle*np.pi/180))
        #                       for i in np.arange(auxtop[-1]+stepz, auxbot[-1]-stepz, stepz)])

# pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]

# with open('/home/hector/Documents/point_cloud/pcd_ring1.data', 'wb') as filehandle:
    # store the data as binary data stream
    # pickle.dump(pts, filehandle)
with open('/home/hector/Documents/point_cloud/pcd_rings.data', 'rb') as filehandle:
    # store the data as binary data stream
    pts = pickle.load(filehandle)
steps = (stepx, stepy, stepz)
print(f'Forming Point Cloud with normals')
# Gerar normais e mesh
pcd = pl2pc(pts, steps, orient_tangent=False, xlen=int(width/stepx), radius_top=6, radius_bot=12)
# o3d.visualization.draw_geometries([pcd], point_show_normal=False)
o3d.visualization.draw_geometries([pcd], point_show_normal=False)

print(f'Generating Mesh')
mesh = p2m(pcd, depth=8, smooth=False)
# mesh.paint_uniform_color(np.array([61, 39, 33])/255)
# mesh = mesh.simplify_quadric_decimation(100000)
# triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
# triangle_clusters = np.asarray(triangle_clusters)
# cluster_n_triangles = np.asarray(cluster_n_triangles)
# cluster_area = np.asarray(cluster_area)
# triangles_to_remove = cluster_n_triangles[triangle_clusters] < 300
# mesh.remove_triangles_by_mask(triangles_to_remove)
# mesh.compute_triangle_normals()
# o3d.io.write_triangle_mesh('/home/hector/Documents/point_cloud/spool_cenpes_4rings2.stl', mesh, print_progress=True)
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)
