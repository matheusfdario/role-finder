import numpy as np
import open3d as o3d
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt


def generateArtificialCylinder(r_out, r_int, angles, y_span):
    half_idx = np.argmin(np.power(angles - 180, 2))

    # Dados da superfície externa:
    out_surf_points_top = np.zeros((3, half_idx, y_span.shape[0]))
    int_surf_points_top = np.zeros((3, half_idx, y_span.shape[0]))
    out_surf_points_bot = np.zeros((3, angles.shape[0] - half_idx, y_span.shape[0]))
    int_surf_points_bot = np.zeros((3, angles.shape[0] - half_idx, y_span.shape[0]))

    for i, y in enumerate(y_span):
        x_out = np.cos(np.deg2rad(angles)) * r_out[:, i]
        z_out = np.sin(np.deg2rad(angles)) * r_out[:, i]
        x_int = np.cos(np.deg2rad(angles)) * r_int[:, i]
        z_int = np.sin(np.deg2rad(angles)) * r_int[:, i]

        # 0 a 180 graus aproximadamente:
        out_surf_points_top[0, :, i] = x_out[:half_idx]
        out_surf_points_top[1, :, i] = y * np.ones_like(x_out[:half_idx])
        out_surf_points_top[2, :, i] = z_out[:half_idx]

        # 180 a 360 graus aproximadamente:
        out_surf_points_bot[0, :, i] = x_out[half_idx:]
        out_surf_points_bot[1, :, i] = y * np.ones_like(x_out[half_idx:])
        out_surf_points_bot[2, :, i] = z_out[half_idx:]

        # 0 a 180 graus aproximadamente:
        int_surf_points_top[0, :, i] = x_int[:half_idx]
        int_surf_points_top[1, :, i] = y * np.ones_like(x_int[:half_idx])
        int_surf_points_top[2, :, i] = z_int[:half_idx]

        # 0 a 180 graus aproximadamente:
        int_surf_points_bot[0, :, i] = x_int[half_idx:]
        int_surf_points_bot[1, :, i] = y * np.ones_like(x_int[half_idx:])
        int_surf_points_bot[2, :, i] = z_int[half_idx:]

    return [out_surf_points_top, out_surf_points_bot], [int_surf_points_top, int_surf_points_bot]


def generateSides(r_out, r_int, angles, y_span, r_discr=1e-1):
    front_side = list()
    back_side = list()

    for ang_idx, angle in enumerate(angles):
        front_radii = np.linspace(start=r_int[ang_idx, 0], stop=r_out[ang_idx, 0] + r_discr, num=150)
        back_radii = np.linspace(start=r_int[ang_idx, -1], stop=r_out[ang_idx, -1] + r_discr, num=150)

        x_front = np.array([np.cos(np.deg2rad(angle)) * r for r in front_radii]).flatten()
        y_front = y_span[0] * np.ones_like(x_front)
        z_front = np.array([np.sin(np.deg2rad(angle)) * r for r in front_radii]).flatten()

        x_back = np.array([np.cos(np.deg2rad(angle)) * r for r in back_radii]).flatten()
        y_back = y_span[-1] * np.ones_like(x_back)
        z_back = np.array([np.sin(np.deg2rad(angle)) * r for r in back_radii]).flatten()

        for j in range(x_front.shape[0]):
            front_side.append((x_front[j], y_front[j], z_front[j]))
            back_side.append((x_back[j], y_back[j], z_back[j]))

    return front_side, back_side


def generate_Cylinder_pcd(r_out, r_int, angles, y_span):
    # Transforma de coordenadas cilindricas para cartesianas:
    out_surf_points, int_surf_points = generateArtificialCylinder(r_out, r_int, angles, y_span)

    # Gera as coordenadas em formato de tupla:
    generate_tuple_coords = lambda xyz: \
        [(xyz[0, i, j], xyz[1, i, j], xyz[2, i, j])
         for i in range(xyz.shape[1])
         for j in range(xyz.shape[2])]
    out_surf1 = generate_tuple_coords(out_surf_points[0])
    out_surf2 = generate_tuple_coords(out_surf_points[1])
    int_surf1 = generate_tuple_coords(int_surf_points[0])
    int_surf2 = generate_tuple_coords(int_surf_points[1])

    # Gera os pontos pertencentes à frente e costas do cilindro:
    front_surf, back_surf = generateSides(r_out, r_int, angles, y_span)

    # Meia cana superior (0 a 180 graus)
    top_out_pcd = o3d.geometry.PointCloud()
    top_out_pcd.points = o3d.utility.Vector3dVector(out_surf1)
    top_out_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=360))
    top_out_pcd.orient_normals_to_align_with_direction()

    # Meia cana inferior (180 a 360 graus)
    bot_out_pcd = o3d.geometry.PointCloud()
    bot_out_pcd.points = o3d.utility.Vector3dVector(out_surf2)
    bot_out_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=360))
    bot_out_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))

    # Meia cana superior (0 a 180 graus)
    top_int_pcd = o3d.geometry.PointCloud()
    top_int_pcd.points = o3d.utility.Vector3dVector(int_surf1)
    top_int_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=360))
    top_int_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))

    # Meia cana inferior (180 a 360 graus)
    bot_int_pcd = o3d.geometry.PointCloud()
    bot_int_pcd.points = o3d.utility.Vector3dVector(int_surf2)
    bot_int_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=360))
    bot_int_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Frente
    front_pcd = o3d.geometry.PointCloud()
    front_pcd.points = o3d.utility.Vector3dVector(front_surf)
    front_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=360))
    front_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, -1, 0]))

    # Fundo
    back_pcd = o3d.geometry.PointCloud()
    back_pcd.points = o3d.utility.Vector3dVector(back_surf)
    back_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=360))
    back_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 1, 0]))

    return top_out_pcd, bot_out_pcd, top_int_pcd, bot_int_pcd, front_pcd, back_pcd


ang_step = 5
angles = np.arange(0, 360, ang_step)
y_span = np.linspace(0, 2 * 24, num=24)
# Gera pontos artificialmente:
# out_surf_points, int_surf_points = generateArtificialCylinder(67, 67 - 16, angles, y_span)

#
n_ang = angles.shape[0]
n_shots = y_span.shape[0]


r_out_exp = np.array([66.01401403, 66.07457381, 66.15809697, 66.24237809, 66.3274219,
                  66.41317716, 66.49952606, 66.58625841, 66.67310135, 66.75975381,
                  66.84592718, 66.93135567, 67.0158069, 67.09909759, 67.18110564,
                  67.26176483, 67.32243715, 67.31207235, 67.22607929, 67.07773043,
                  66.88260704, 66.67305835, 66.46239534, 66.25064759, 66.03784472,
                  65.82402002, 65.6092069, 65.39343506, 65.17672557, 64.95908687,
                  64.74453844, 64.55385051, 64.42236386, 64.36169905, 64.35515221,
                  64.35703864, 64.3579721])

r_out = np.concatenate((r_out_exp[:-1], r_out_exp[1:][::-1]))
r_out = np.tile(r_out, (n_shots, 1)).transpose()

r_int_exp = np.array([50.62727264, 50.65421456, 50.70883027, 50.78944517, 50.88315596,
                  50.95715905, 51.08639797, 51.30710025, 51.59931707, 51.87777358,
                  52.06652463, 52.16460193, 52.22418572, 52.32778114, 52.56473797,
                  52.86671933, 53.15369702, 53.30628376, 53.25672819, 53.04038537,
                  52.70568037, 52.37988022, 52.18479398, 52.15141586, 52.17822355,
                  52.17551428, 52.07690922, 51.85571629, 51.54024713, 51.23934789,
                  50.98426277, 50.73324363, 50.51386342, 50.35467941, 50.25296802,
                  50.19717519, 50.17285085])

r_int = np.concatenate((r_int_exp[:-1], r_int_exp[1:][::-1]))
r_int = np.tile(r_int, (n_shots, 1)).transpose()


r_out = np.load('ext_surf_mat.npy')
r_int = np.load('int_surf_mat.npy')

# Cria nuvem de ponto:
pcd_list = generate_Cylinder_pcd(r_out, r_int, angles, y_span)
pcd = np.sum(pcd_list[:])

# Plota alguns dos pcds:
o3d.visualization.draw_geometries(pcd_list, mesh_show_back_face=True, mesh_show_wireframe=False)

# Cria os mesh:
print("Meshing...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=9)

print("Done")
mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))

# PLota o resultado 3D:
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)

# 3400 velocoidade do acrílico tewstar smsartwedge
