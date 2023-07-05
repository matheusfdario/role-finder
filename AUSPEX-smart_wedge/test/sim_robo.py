import numpy as np
from framework import post_proc, pre_proc, file_civa
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
from scipy.spatial.transform import Rotation as R
import pickle

## Editar para mudar os arquivos lidos
# Editar para trocar .var
base_path = '/home/hector/PycharmProjects/AUSPEX/data/sim_aq_robo_fmc.var/proc0/results/Config_['
end_path = ']/model.civa'
lista = [base_path+str(i)+end_path for i in range(12)]
angles = -np.linspace(10, 340, 12)

# Define se carrega os dados salvos, se calcula e, se nesse caso, salva os dados calculados.
read_data = False
write_data = False
## Parâmetros para criacão do modelo 3D. Os steps definem o tamanho da grade
step = 1
stepx = 0.1
stepy = 0.1
stepz = 0.1
steps = (stepx, stepy, stepz)

# Define tamanho da ROI
width = 28
height = 25.0
corner_roi = np.array([-width / 2, 0.0, 25.0])[np.newaxis, :]
roi = ImagingROI(corner_roi, height=height, width=width, h_len=int(height / stepz), w_len=int(width / stepx),
                 depth=1.0, d_len=1)

surftop = []
surfbot = []
pfac1 = []
pfac2 = []
psid1 = []
psid2 = []

if read_data:
    print('Loading')
    with open('/home/hector/PycharmProjects/AUSPEX/data/sim_robo.data', 'rb') as filehandle:
        # store the data as binary data stream
        pts = pickle.load(filehandle)
else:
## Laços para reconstruções
    for i, file in enumerate(lista): # Itera sobre os arquivos da lista
        print(file)
        datas = file_civa.read(file, read_ascan=False)
        shots = datas.inspection_params.step_points.shape[0]
        if i == 0:
            ref = np.asarray([0, 0, -datas.inspection_params.step_points[0][2]])
        r = R.from_euler('y', angles[i], degrees=True)
        shots = 2
        for j in range(shots):
            print(f'Step: {j} - {j+i*shots}/{shots*len(lista)} - {(j+i*shots)/(shots*len(lista))*100:.2f}%',
                  end='\r', flush=True)
            data = file_civa.read(file, sel_shots=j)
            _ = pre_proc.hilbert_transforms(data)

            if j > 0:
                bot_ant = bot
                top_ant = top
                coord_ant = np.copy(coord)
            coord = datas.inspection_params.step_points[j] + ref
            surfx = roi.w_points
            data.surf = Surface(data, surf_type=SurfaceType.CIRCLE_NEWTON)
            data.surf.fit()

            chave = tfm.tfm_kernel(data, roi=roi, sel_shot=0)
            #chave = tfmcuda.tfmcuda_kernel(data, roi=roi, sel_shot=0)

            yt = post_proc.envelope(data.imaging_results[chave].image)
            if yt.sum() > 0:
                y = yt/yt.max()

            # Surface Estimation
            a = img_line(y)
            z = roi.h_points[a[0].astype(int)]
            w = a[1]
            lamb = 1
            rho = 100
            bot, _, _, _, _ = intsurf_estimation.profile_fadmm(w.ravel(), z, lamb, x0=z, rho=rho, eta=.999,
                                                                     itmax=250, tol=1e-9, max_iter_cg=1500)
            top = np.interp(surfx, data.surf.x_discr, data.surf.z_discr)
            # print(f'Making points')
            # Os pontos são rotacionados de acordo com a orientacão do transdutor usando o operador Rotation
            ptop = r.apply([(surfx[k]+coord[0], coord[1], top[k]-coord[2]) for k, _ in enumerate(surfx)])
            surftop.extend(ptop)
            pbot = r.apply([(surfx[k]+coord[0], coord[1], bot[k]-coord[2]) for k, _ in enumerate(surfx)])
            surfbot.extend(pbot)
            L = int(0.5/stepy)
            if j > 0:
                for l in range(L - 1):
                    auxtop = (l + 1) / L * (top - top_ant) + top_ant
                    auxbot = (l + 1) / L * (bot - bot_ant) + bot_ant
                    auxy = (l + 1) / L * (coord[1] - coord_ant[1]) + coord_ant[1]
                    surftop.extend(r.apply([(surfx[m]+coord[0], auxy, auxtop[m]-coord[2])
                                            for m, _ in enumerate(surfx)]))
                    surfbot.extend(r.apply([(surfx[m]+coord[0], auxy, auxbot[m]-coord[2])
                                            for m, _ in enumerate(surfx)]))
            for l, wp in enumerate(surfx):
                if j == 0:
                    pfac1.extend(r.apply([(surfx[l]+coord[0], coord[1], m-coord[2]) for m in np.arange(top[l]+stepz, bot[l]-stepz, stepz)]))
                if j == shots-1:
                    pfac2.extend(r.apply([(surfx[l]+coord[0], coord[1], m-coord[2]) for m in np.arange(top[l]+stepz, bot[l]-stepz, stepz)]))
    print(f'Step: Done - {len(lista)*shots}/{shots*len(lista)} - 100%', end='\r', flush=True)
    print('')

    pts = [surftop, surfbot, pfac1, pfac2, psid1, psid2]

    if write_data:
        print('Saving')
        with open('/home/hector/PycharmProjects/AUSPEX/data/sim_robo.data', 'wb') as filehandle:
                # store the data as binary data stream
                pickle.dump(pts, filehandle)


# Gerar normais e mesh
print('Estimating normals')
pcd = pl2pc(pts, steps, orient_tangent=True, xlen=roi.w_len)
# o3d.io.write_point_cloud('/home/hector/Desktop/teste.ply', pcd, print_progress=True)
# pcd = o3d.io.read_point_cloud('/home/hector/PycharmProjects/AUSPEX/data/sim_robo.ply')
##
# Na janela de visualizacão, pode-se apertar a tecla 'n' para alternar a visualizacão das normais.
# Apertando '-' e '+' tambem controlam o tamanho dos vetores.
o3d.visualization.draw_geometries([pcd])

mesh = p2m(pcd, depth=8)
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=True)
print('Simplifying mesh')
mesh2 = mesh.simplify_quadric_decimation(5000)
mesh2 = mesh2.simplify_vertex_clustering(0.1)
mesh2.compute_triangle_normals()
mesh2.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh2], mesh_show_back_face=True, mesh_show_wireframe=True)
# o3d.io.write_triangle_mesh('/home/hector/Desktop/teste.stl', mesh, print_progress=True)
