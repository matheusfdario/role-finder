import numpy as np
import open3d as o3d
from framework.utils import pointlist_to_cloud as pl2pc
from framework.utils import pcd_to_mesh as p2m
from matplotlib import pyplot as plt

angles1 = np.arange(0, 180 + 2, 2)
angles2 = np.arange(180, 360 + 2, 2)
angles3 = np.arange(0, 360 + 2, 2)
int_radius = 67 - 16
out_radius = 67

# Supondo uma superfície ideal:
out_x1 = np.cos(np.deg2rad(angles1)) * out_radius
out_z1 = np.sin(np.deg2rad(angles1)) * out_radius
int_x1 = np.cos(np.deg2rad(angles1)) * int_radius
int_z1 = np.sin(np.deg2rad(angles1)) * int_radius
out_x2 = np.cos(np.deg2rad(angles2)) * out_radius
out_z2 = np.sin(np.deg2rad(angles2)) * out_radius
int_x2 = np.cos(np.deg2rad(angles2)) * int_radius
int_z2 = np.sin(np.deg2rad(angles2)) * int_radius

y_position = np.arange(0, 50 + 1e-1, 1e-1)

out_surf1 = list()
int_surf1 = list()
out_surf2 = list()
int_surf2 = list()
front_surf = list()
back_surf = list()

side_radius = np.arange(int_radius, out_radius, 1e-1)
side_x = np.array([np.cos(np.deg2rad(angles3)) * radius for radius in side_radius]).flatten()
side_z = np.array([np.sin(np.deg2rad(angles3)) * radius for radius in side_radius]).flatten()
front_y = np.ones_like(side_x) * y_position[0]
back_y = np.ones_like(side_x) * y_position[-1]

for y in y_position:
    slice_out_coord = np.array([out_x1, y * np.ones_like(out_x1), out_z1], dtype=float)
    temp_vec = [(slice_out_coord[0, i], slice_out_coord[1, i], slice_out_coord[2, i]) for i in range(slice_out_coord.shape[1])]
    for vec in temp_vec:
        out_surf1.append(vec)

    slice_int_coord = np.array([int_x1, y * np.ones_like(int_x1), int_z1], dtype=float)
    temp_vec = [(slice_int_coord[0, i], slice_int_coord[1, i], slice_int_coord[2, i]) for i in
                range(slice_int_coord.shape[1])]
    for vec in temp_vec:
        int_surf1.append(vec)

    slice_out_coord = np.array([out_x2, y * np.ones_like(out_x2), out_z2], dtype=float)
    temp_vec = [(slice_out_coord[0, i], slice_out_coord[1, i], slice_out_coord[2, i]) for i in range(slice_out_coord.shape[1])]
    for vec in temp_vec:
        out_surf2.append(vec)

    slice_int_coord = np.array([int_x2, y * np.ones_like(int_x2), int_z2], dtype=float)
    temp_vec = [(slice_int_coord[0, i], slice_int_coord[1, i], slice_int_coord[2, i]) for i in
                range(slice_int_coord.shape[1])]
    for vec in temp_vec:
        int_surf2.append(vec)

front_surf = [(side_x[i], y_position[0], side_z[i]) for i in range(0, side_x.shape[0])]
back_surf = [(side_x[i], y_position[-1], side_z[i]) for i in range(0, side_x.shape[0])]


# Meia cana superior (0 a 180 graus)
top_out_pcd = o3d.geometry.PointCloud()
top_out_pcd.points = o3d.utility.Vector3dVector(out_surf1)
top_out_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=60))
top_out_pcd.orient_normals_to_align_with_direction()


# Meia cana inferior (180 a 360 graus)
bot_out_pcd = o3d.geometry.PointCloud()
bot_out_pcd.points = o3d.utility.Vector3dVector(out_surf2)
bot_out_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=60))
bot_out_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))


# Meia cana superior (0 a 180 graus)
top_int_pcd = o3d.geometry.PointCloud()
top_int_pcd.points = o3d.utility.Vector3dVector(int_surf1)
top_int_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=60))
top_int_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, -1]))


# Meia cana inferior (180 a 360 graus)
bot_int_pcd = o3d.geometry.PointCloud()
bot_int_pcd.points = o3d.utility.Vector3dVector(int_surf2)
bot_int_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=60))
bot_int_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

# Frente
front_pcd = o3d.geometry.PointCloud()
front_pcd.points = o3d.utility.Vector3dVector(front_surf)
front_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=60))
front_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, -1, 0]))

# FUndo
back_pcd = o3d.geometry.PointCloud()
back_pcd.points = o3d.utility.Vector3dVector(back_surf)
back_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=80, max_nn=60))
back_pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 1, 0]))

pcd = [top_out_pcd + bot_out_pcd + top_int_pcd + bot_int_pcd + front_pcd + back_pcd]

radius=50
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                           o3d.utility.DoubleVector([radius, radius*2, radius*0.5]))
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)
mesh.paint_uniform_color(np.array([0.5, 0.5, 0.5]))
mesh = mesh.simplify_quadric_decimation(100000)
mesh.compute_triangle_normals()
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, mesh_show_wireframe=False)




