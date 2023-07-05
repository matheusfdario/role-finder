import numpy as np
from matplotlib import pyplot as plt
import time
from framework import file_m2k, file_civa, post_proc, pre_proc
from framework.data_types import ImagingROI
from imaging import tfm
from surface.surface import SurfaceType
from surface.surface import Surface
from parameter_estimation import intsurf_estimation
from parameter_estimation.intsurf_estimation import img_line
# import pyvista as pv
# import pymeshfix as mf
import open3d as o3d

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
l = 0

for k, file in enumerate(lista):
    print(file)
    data = file_m2k.read(file,
                         freq_transd=5, bw_transd=0.5, tp_transd='gaussian', sel_shots=0)
    _ = pre_proc.hilbert_transforms(data)

    corner_roi = np.array([-14.0, 0.0, 15.0])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=10.0, width=28.0, h_len=400, w_len=400, depth=1.0, d_len=1)
    surfx = roi.w_points
    data.inspection_params.type_insp = 'immersion'
    data.inspection_params.coupling_cl = 1483.0
    data.surf = Surface(data, surf_type=SurfaceType.CIRCLE_NEWTON)
    data.surf.fit()

    t0 = time.time()
    chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0, c=5885.5)
    print(time.time()-t0)

    yt = post_proc.envelope(data.imaging_results[chave].image)
    y = yt/yt.max()

    # Surface Estimation
    a = img_line(y)
    z = roi.h_points[a[0].astype(int)]
    w = a[1]
    lamb = 10
    rho = 100
    print(f'Estimating Surface')
    t0 = time.time()
    bot, resf, kf, pk, sk = intsurf_estimation.profile_fadmm(w.ravel(), z, lamb, x0=z, rho=rho, eta=.999,
                                                             itmax=250, tol=1e-9, max_iter_cg=500)
    print(time.time()-t0)
    top = np.interp(surfx, data.surf.x_discr, data.surf.z_discr)
    try:
        auxtop += top
    except NameError:
        auxtop = top
    surftop.extend([(surfx[i], top[i], surfy[k]) for i in range(len(surfx))])
    if k == 0:
        surftop.extend([(surfx[i], top[i], surfy[k]+0.5) for i in range(len(surfx))])
        surftop.extend([(surfx[i], top[i], surfy[k]+1.5) for i in range(len(surfx))])
    else:
        surftop.extend([(surfx[i], top[i], surfy[k]-0.5) for i in range(len(surfx))])
        surftop.extend([(surfx[i], top[i], surfy[k]-1.5) for i in range(len(surfx))])

    surfbot.extend([(surfx[i], bot[i], surfy[k]) for i in range(len(surfx))])
    try:
        auxbot += bot
    except NameError:
        auxbot = bot
    surfbot.extend([(surfx[i], bot[i], surfy[k]) for i in range(len(surfx))])
    if k == 0:
        surfbot.extend([(surfx[i], bot[i], surfy[k]+0.5) for i in range(len(surfx))])
        surfbot.extend([(surfx[i], bot[i], surfy[k]+1.5) for i in range(len(surfx))])
    else:
        surfbot.extend([(surfx[i], bot[i], surfy[k]-0.5) for i in range(len(surfx))])
        surfbot.extend([(surfx[i], bot[i], surfy[k]-1.5) for i in range(len(surfx))])
    for j, wp in enumerate(surfx):
        if k == 0:
            pfac1.extend([(surfx[j], i, surfy[k]) for i in np.arange(top[j]+0.1, bot[j]-0.1, 0.1)])
        else:
            pfac2.extend([(surfx[j], i, surfy[k]) for i in np.arange(top[j]+0.1, bot[j]-0.1, 0.1)])
    # surftop.append(list(np.interp(surfx, data.surf.x_discr, data.surf.z_discr)))
    # surfbot.append(list(zf))
auxtop /= (k+1)
auxbot /= (k+1)
psid1.extend([(surfx[0], i, surfy[0]+0.5) for i in np.arange(top[0]+0.2, bot[0]-0.2, 1)])
psid1.extend([(surfx[0], i, surfy[0]+1.5) for i in np.arange(top[0]+0.2, bot[0]-0.2, 1)])
psid1.extend([(surfx[0], i, surfy[1]-0.5) for i in np.arange(top[0]+0.2, bot[0]-0.2, 1)])
psid1.extend([(surfx[0], i, surfy[1]-1.5) for i in np.arange(top[0]+0.2, bot[0]-0.2, 1)])
psid1.extend([(surfx[0], i, np.mean(surfy)) for i in np.arange(top[0]+0.2, bot[0]-0.2, 1)])
psid2.extend([(surfx[-1], i, surfy[0]+0.5) for i in np.arange(top[-1]+0.2, bot[-1]-0.2, 1)])
psid2.extend([(surfx[-1], i, surfy[0]+1.5) for i in np.arange(top[-1]+0.2, bot[-1]-0.2, 1)])
psid2.extend([(surfx[-1], i, surfy[1]-0.5) for i in np.arange(top[-1]+0.2, bot[-1]-0.2, 1)])
psid2.extend([(surfx[-1], i, surfy[1]-1.5) for i in np.arange(top[-1]+0.2, bot[-1]-0.2, 1)])
psid2.extend([(surfx[-1], i, np.mean(surfy)) for i in np.arange(top[-1]+0.2, bot[-1], 1)])
surftop.extend([(surfx[i], auxtop[i], np.mean(surfy)) for i in range(len(surfx))])
surfbot.extend([(surfx[i], auxbot[i], np.mean(surfy)) for i in range(len(surfx))])

# Gerar normais e mesh
pcdtop = o3d.geometry.PointCloud()
pcdtop.points = o3d.utility.Vector3dVector(surftop)
pcdtop.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=5*2))
pcdtop.orient_normals_consistent_tangent_plane(int(np.rint(step*2)))
pcdtop.normals = o3d.utility.Vector3dVector(-(np.asarray(pcdtop.normals)))

pcdbot = o3d.geometry.PointCloud()
pcdbot.points = o3d.utility.Vector3dVector(surfbot)
pcdbot.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=5*2))
pcdbot.orient_normals_consistent_tangent_plane(int(np.rint(step*2)))
pcdbot.normals = o3d.utility.Vector3dVector((np.asarray(pcdbot.normals)))

pcdf1 = o3d.geometry.PointCloud()
pcdf1.points = o3d.utility.Vector3dVector(pfac1)
pcdf1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=step*2))
pcdf1.normals = o3d.utility.Vector3dVector(np.tile([0, 0, -1],
                                                    [np.asarray(pcdf1.normals).shape[0],1]))

pcdf2 = o3d.geometry.PointCloud()
pcdf2.points = o3d.utility.Vector3dVector(pfac2)
pcdf2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=step*2))
pcdf2.normals = o3d.utility.Vector3dVector(np.tile([0, 0, 1],
                                                    [np.asarray(pcdf2.normals).shape[0],1]))

pcds1 = o3d.geometry.PointCloud()
pcds1.points = o3d.utility.Vector3dVector(psid1)
pcds1.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=2*step))
pcds1.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([-1, 0, 0])),
                                                    [np.asarray(pcds1.normals).shape[0],1]))

pcds2 = o3d.geometry.PointCloud()
pcds2.points = o3d.utility.Vector3dVector(psid2)
pcds2.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamRadius(radius=2*step))
pcds2.normals = o3d.utility.Vector3dVector(np.tile((np.asarray([1, 0, 0])),
                                                    [np.asarray(pcds2.normals).shape[0],1]))

print('Meshing')
points = pcdbot + pcdtop + pcdf1 + pcdf2 + pcds1 + pcds2
t0 = time.time()
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(points, depth=7, scale=1.4, linear_fit=True)[0]
print(time.time()-t0)
# mesh = mesh.crop(box)
print(f'Generated mesh with {len(mesh.triangles)} triangles')
mesh.paint_uniform_color(np.array([0.5,0.5,0.5]))
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
mesh2 = mesh.filter_smooth_laplacian()
mesh2.compute_triangle_normals()
mesh2.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh2], mesh_show_back_face=True, mesh_show_wireframe=False)
# o3d.io.write_triangle_mesh('/home/hector/Documents/point_cloud/cenpes.stl', mesh2, print_progress=True)
# mesh.asdaw()
# v = np.asarray(mesh.vertices)
# f = np.array(mesh.triangles)
# f = np.c_[np.full(len(f), 3), f]
# meshpv = pv.PolyData(v, f)
# print('Repairing Mesh')
# meshfix = mf.MeshFix(meshpv)
# meshfix.repair(verbose=False)
# meshpv = meshfix.mesh
# meshpv.smooth()
# print('Decimating')
# target_faces = 50000
# ratio = 1 - target_faces/meshpv.number_of_faces
# meshpv2 = meshpv.decimate_pro(ratio, preserve_topology=True, inplace=False)
# print(f'Simplified to {meshpv.number_of_faces} triangles')
# meshpv2.compute_normals()
# meshpv2.plot()
# pv.save_meshio('/home/hector/Documents/point_cloud/cenpes.obj', meshpv2)