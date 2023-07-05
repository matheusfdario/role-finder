import numpy as np
import datetime
from framework import file_m2k, post_proc, pre_proc, file_civa
from framework.data_types import ImagingROI
from imaging import tfm
# from imaging import tfmcuda
from surface.surface import SurfaceType
from surface.surface import Surface
from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line
import open3d as o3d
from framework.utils import p2c
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
import pickle

print("Starting time:-", datetime.datetime.now())

## Editar para mudar os arquivos lidos
# Editar para trocar .var
path = lambda x: f'/home/hector/PycharmProjects/AUSPEX/data/acq_robo_25_05/acq_robo_s2_{x:02}.m2k'
lista = [path(i) for i in range(1, 25)]
angles = -np.arange(24)*360/24-10
read_data = False
write_data = False
## Parâmetros para reconstrução
step = 1
stepx = 0.1
stepy = 0.1
stepz = 0.1
steps = (stepx, stepy, stepz)

surftop = []
surfbot = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []

pcd = o3d.geometry.PointCloud()


if read_data:
    print('Loading')
    with open('/home/hector/Public/acq_robo.data', 'rb') as filehandle:
        # store the data as binary data stream
        pts = pickle.load(filehandle)
else:
## Laços para reconstruções
    for i, file in enumerate(lista): # Itera sobre os arquivos da lista
        print(file)
        datas = file_m2k.read(file, 5, 0.5, 'gaussian', read_ascan=False)
        datas.inspection_params.step_points = datas.inspection_params.step_points[:, [1, 0, 2]]
        shots = datas.inspection_params.step_points.shape[0]
        shots = 40
        if i == 0:
            ref = np.asarray([0, 0, -datas.inspection_params.step_points[0][2]])
        r = R.from_euler('y', angles[i], degrees=True)
        for j in range(shots):
            print(f'Step: {j} - {j+i*shots}/{shots*len(lista)} - {(j+i*shots)/(shots*len(lista))*100:.2f}%',
                  end='\r', flush=True)
            data = file_m2k.read(file, 5, 0.5, 'gaussian', sel_shots=j)
            _ = pre_proc.hilbert_transforms(data)

            if j > 0:
                bot_ant = bot
                top_ant = top
                coord_ant = np.copy(coord)
            # print(coord)
            coord = datas.inspection_params.step_points[j] + ref

            data.inspection_params.type_insp = 'immersion'
            data.inspection_params.coupling_cl = 1483.0
            data.surf = Surface(data, surf_type=SurfaceType.CIRCLE_QUAD)
            data.surf.fit()

            cond_radius = np.abs(data.surf.surfaceparam.r - 70) < 5
            cond_center_x = (np.abs(data.surf.surfaceparam.x) < 1)
            cond_waterpath = (data.surf.surfaceparam.water_path > 5) and (data.surf.surfaceparam.water_path < 25)
            if cond_radius and cond_waterpath:
                waterpath = data.surf.surfaceparam.water_path

                if j == 0 and i == 0:
                    wp_ref = waterpath
                    wp = 0
                else:
                    # wp = wp_ref-np.asarray([0.0, 0.0, waterpath])
                    wp = wp_ref - waterpath
                rad = 70.0 - wp
                dx = -rad*np.sin(np.pi*angles[i]/180)
                dz = -rad*np.cos(np.pi*angles[i]/180)
                shift = np.asarray([dx, 0.0, dz])
                width = 7
                height = 25.0
                corner_roi = np.array([-width / 2, 0.0, waterpath+5])[np.newaxis, :]
                roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height / stepz),
                                 w_len=int(width / stepx),
                                 depth=1.0, d_len=1)

                # t0 = time.time()
                chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0)
                # print(time.time()-t0)

                yt = post_proc.envelope(data.imaging_results[chave].image)
                if yt.sum() > 0:
                    y = yt/yt.max()

                # Surface Estimation
                a = img_line(y)
                z = roi.h_points[a[0].astype(int)]
                w = a[1]
                lamb = 0.5
                rho = 100
                # print(f'Estimating Surface')
                # t0 = time.time()
                bot, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w.ravel(), z, lamb, x0=z, rho=rho, eta=.999,
                                                                         itmax=250, tol=1e-9, max_iter_cg=1500)
                # print(time.time()-t0)

                surfx = roi.w_points
                surfxt = np.linspace(-6, 6, 120)
                top = np.interp(surfxt, data.surf.x_discr, data.surf.z_discr)
            else:
                print(f"File: {file}\nShot: {j}\nCondition radius: {cond_radius}\nCondition waterpath: {cond_waterpath}")
                print(f"Radius: {data.surf.surfaceparam.r}\nWaterpath: {data.surf.surfaceparam.water_path}")
                dx = -rad*np.sin(np.pi*angles[i]/180)
                dz = -rad*np.cos(np.pi*angles[i]/180)
                shift = np.asarray([dx, 0.0, dz])
                top = top_ant
                bot = bot_ant
            # print(f'Making points')
            ptop = r.apply([(surfxt[k]+coord[0], coord[1], top[k]-coord[2]) for k, _ in enumerate(surfxt)]) + shift
            surftop.extend(ptop)
            pbot = r.apply([(surfx[k]+coord[0], coord[1], bot[k]-coord[2]) for k, _ in enumerate(surfx)]) + shift
            surfbot.extend(pbot)
            L = int(0.5/stepy)
            if j > 0:
                for l in range(L - 1):
                    auxtop = (l + 1) / L * (top - top_ant) + top_ant
                    auxbot = (l + 1) / L * (bot - bot_ant) + bot_ant
                    auxy = (l + 1) / L * (coord[1] - coord_ant[1]) + coord_ant[1]
                    surftop.extend(r.apply([(surfxt[m]+coord[0], auxy, auxtop[m]-coord[2])
                                            for m, _ in enumerate(surfxt)])+shift)
                    surfbot.extend(r.apply([(surfx[m]+coord[0], auxy, auxbot[m]-coord[2])
                                            for m, _ in enumerate(surfx)])+shift)
            for l, wp in enumerate(surfx):
                if j == 0:
                    pfac1.extend(r.apply([(surfx[l]+coord[0], coord[1], m-coord[2])
                                          for m in np.arange(top[l]+stepz, bot[l]-stepz, stepz)])+shift)
                if j == shots-1:
                    pfac2.extend(r.apply([(surfx[l]+coord[0], coord[1], m-coord[2])
                                          for m in np.arange(top[l]+stepz, bot[l]-stepz, stepz)])+shift)
            if j>0 and j%2 == 0:
                pcd += p2c(surftop, outer_surf=True, angle=angles[i])
                pcd += p2c(surfbot, outer_surf=False, angle=angles[i])
                pcd += p2c(pfac1, force_direction=np.asarray([0.0, -1.0, 0.0]))
                pcd += p2c(pfac2, force_direction=np.asarray([0.0, 1.0, 0.0]))
                surftop = []
                surfbot = []
                pfac1 = []
                pfac2 = []
        if len(surftop)>0:
            pcd += p2c(surftop, outer_surf=True, angle=angles[i])
            pcd += p2c(surfbot, outer_surf=False, angle=angles[i])
            pcd += p2c(pfac1, force_direction=np.asarray([0.0, -1.0, 0.0]))
            pcd += p2c(pfac2, force_direction=np.asarray([0.0, 1.0, 0.0]))
            surftop = []
            surfbot = []
            pfac1 = []
            pfac2 = []

    print(f'Step: Done - {len(lista)*shots}/{shots*len(lista)} - 100%', end='\r', flush=True)
    print('')

    # pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]

    # if write_data:
    #     print('Saving')
    #     with open('/home/hector/Public/acq_robo.data', 'wb') as filehandle:
    #             # store the data as binary data stream
    #             pickle.dump(pts, filehandle)


# Gerar normais e mesh
print('Estimating normals')
# pcd = pl2pc(pts, steps, orient_tangent=True, xlen=roi.w_len)
o3d.io.write_point_cloud('/home/hector/Public/acq_robo_rot.ply', pcd, print_progress=True)
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# print('Meshing')
print("Pre meshing time:-", datetime.datetime.now())
mesh = p2m(pcd, depth=8, scale=1.1)
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
# # print('Simplifying mesh')
# mesh2 = mesh.simplify_quadric_decimation(50000)
# mesh2 = mesh2.simplify_vertex_clustering(0.1)
# mesh2.compute_triangle_normals()
# mesh2.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh2], mesh_show_back_face=True, mesh_show_wireframe=True)
o3d.io.write_triangle_mesh('/home/hector/Public/acq_robo_rot.stl', mesh, print_progress=True)

print("Finished time:-", datetime.datetime.now())