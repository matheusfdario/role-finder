import numpy as np
import open3d as o3d
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt

lista = np.linspace(0, 360, 457)
surftop = []
surfbot = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []
radius = 70
waterpath = 10
surfy = np.sin(np.array(lista)*np.pi/180)
step = 1
stepx = 0.1
stepy = 0.1
stepz = 0.1
l = 0

width = 28.0
height = 20.0

print(f'Making points')
surfx = np.linspace(-20, 20, 400)
for k, angle in enumerate(lista):
    if k>0:
        bot_ant = bot
        top_ant = top
    top = 10*np.ones(400)
    bot = 30*np.ones(400)+5*np.random.random(400)
    surftop.extend([(surfx[i], (radius+waterpath-top[i])*np.cos(angle*np.pi/180), (radius+waterpath-top[i])*np.sin(angle*np.pi/180))
                    for i in range(len(surfx))])
    surfbot.extend([(surfx[i], (radius+waterpath-bot[i])*np.cos(angle*np.pi/180), (radius+waterpath-bot[i])*surfy[k]) for i in range(len(surfx))])
    L = 2
    if k > 0:
        for l in range(L - 1):
            auxtop = (l + 1) / L * (top - top_ant) + top_ant
            auxbot = (l + 1) / L * (bot - bot_ant) + bot_ant
            auxy = (l + 1) / L * (surfy[k] - surfy[k - 1]) + surfy[k - 1]
            # surftop.extend([(surfx[i], auxtop[i], auxy) for i in range(len(surfx))])
            # surfbot.extend([(surfx[i], auxbot[i], auxy) for i in range(len(surfx))])
            psid1.extend([(surfx[0], (radius+waterpath-i)*np.cos(angle*np.pi/180), (radius+waterpath-i)*np.sin(angle*np.pi/180))
                          for i in np.arange(auxtop[0]+stepz, auxbot[0]-stepz, stepz)])
            psid2.extend([(surfx[-1], (radius+waterpath-i)*np.cos(angle*np.pi/180), (radius+waterpath-i)*np.sin(angle*np.pi/180))
                          for i in np.arange(auxtop[-1]+stepz, auxbot[-1]-stepz, stepz)])
    # for j, wp in enumerate(surfx):
    #     if k == 0:
    #         pfac1.extend([(surfx[j], i, surfy[k]) for i in np.arange(top[j]+stepz, bot[j]-stepz, stepz)])
    #     if k == len(lista)-1:
    #         pfac2.extend([(surfx[j], i, surfy[k]) for i in np.arange(top[j]+stepz, bot[j]-stepz, stepz)])

print(f'Forming Point Cloud with normals')
pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]
steps = (stepx, 0.1, stepz)
# Gerar normais e mesh
pcd = pl2pc(pts, steps, xlen=len(surfx))
# o3d.visualization.draw_geometries([pcd], point_show_normal=True)
mesh = p2m(pcd, depth=8)
mesh = mesh.simplify_quadric_decimation(100000)
mesh.compute_triangle_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)
# o3d.io.write_triangle_mesh('/home/hector/Documents/point_cloud/cenpes.stl', mesh, print_progress=True)
