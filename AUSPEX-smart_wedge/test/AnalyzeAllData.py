import matplotlib.pyplot as plt
import numpy as np
import time
from surface.surface import Surface, SurfaceType
from framework import file_m2k, file_mat, file_civa, file_omniscan
from scipy.signal import hilbert
from imaging import tfm
from framework.data_types import ImagingROI, ImagingROI3D, ElementGeometry
from framework import file_m2k, file_civa, post_proc
import sys
import gc


xdczerototal = -1

db3 = 1/2
db6 = db3/(2)
db9 = db6/(2)

corner_roi = np.array([-15.0, -15, 15.0])[np.newaxis, :]
roi = ImagingROI3D(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=61)

sizes = np.zeros(13)
pixels_3dB = np.zeros((13, 7))
pixels_6dB = np.zeros((13, 7))
pixels_9dB = np.zeros((13, 7))
max_pixel = np.zeros((13, 7))
images = list()
images.append(0)

i = 0
for i_config in range(2,28,2):
    radius = i_config/20
    rect_rect_size = i_config/10
    rect_circ_radius = i_config / 20 + 0.1
    hexag_circ_radius = i_config / 20

    sizes[i] = rect_rect_size

    radius = "{:.2f}".format(radius)
    if radius[-1] == '0':
        radius = radius[:-1]
    rect_rect_size = "{:.2f}".format(rect_rect_size)
    if rect_rect_size[-1] == '0':
        rect_rect_size = rect_rect_size[:-1]
    rect_circ_radius = "{:.2f}".format(rect_circ_radius)
    if rect_circ_radius[-1] == '0':
        rect_circ_radius = rect_circ_radius[:-1]
    hexag_circ_radius = "{:.2f}".format(hexag_circ_radius)
    if hexag_circ_radius[-1] == '0':
        hexag_circ_radius = hexag_circ_radius[:-1]




    ret_ret_180_path = './Rect Rect/Angle180/RectRect_pitch'+rect_rect_size+'mm_Angle180.npy'
    print(ret_ret_180_path)
    ret_ret_12_path = './Rect Rect/Angle012/RectRect_pitch' + rect_rect_size + 'mm_Angle012.npy'
    print(ret_ret_12_path)
    ret_circ_180_path = './Rect Circ/Angle180/RectCirc_pitch' + rect_circ_radius + 'mm_Angle180.npy'
    print(ret_circ_180_path)
    ret_circ_12_path = './Rect Circ/Angle012/RectCirc_pitch' + rect_circ_radius + 'mm_Angle012.npy'
    print(ret_circ_12_path)
    hexag_circ_180_path = './Hexag Circ/Angle180/HexagCirc_pitch' + hexag_circ_radius + 'mm_Angle180.npy'
    print(hexag_circ_180_path)
    hexag_circ_12_path = './Hexag Circ/Angle012/HexagCirc_pitch' + hexag_circ_radius + 'mm_Angle012.npy'
    print(hexag_circ_12_path)

    ret_ret_180 = np.load(ret_ret_180_path)
    ret_ret_12 = np.load(ret_ret_12_path)
    ret_circ_180 = np.load(ret_circ_180_path)
    ret_circ_12 = np.load(ret_circ_12_path)
    hexag_circ_180 = np.load(hexag_circ_180_path)
    hexag_circ_12 = np.load(hexag_circ_12_path)

    fig = plt.figure(figsize=(16.0, 10.0))
    # First row: scattering_angle = 180
    # Second row: scattering_angle = 12
    # Columns: ret-ret; ret-circ; hex-circ

    plt.subplot(2, 3, 1)
    envel = post_proc.envelope(ret_ret_180, 0)
    pixels_3dB[i, 1] = np.count_nonzero(envel >= np.max(envel) * db3)
    pixels_6dB[i, 1] = np.count_nonzero(envel >= np.max(envel) * db6)
    pixels_9dB[i, 1] = np.count_nonzero(envel >= np.max(envel) * db9)
    max_pixel[i, 1] = np.max(envel)
    plt.imshow(envel[:,30,:], aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]],
               vmin=0, vmax=0.0185)
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('Rectangle/Rectangle scattering_angle=180')


    plt.subplot(2, 3, 2)
    envel = post_proc.envelope(ret_circ_180, 0)
    pixels_3dB[i, 2] = np.count_nonzero(envel >= np.max(envel) * db3)
    pixels_6dB[i, 2] = np.count_nonzero(envel >= np.max(envel) * db6)
    pixels_9dB[i, 2] = np.count_nonzero(envel >= np.max(envel) * db9)
    max_pixel[i, 2] = np.max(envel)
    plt.imshow(envel[:,30,:], aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]],
               vmin=0, vmax=0.0185)
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('Rectangle/Circle scattering_angle=180')

    plt.subplot(2, 3, 3)
    envel = post_proc.envelope(hexag_circ_180, 0)
    pixels_3dB[i, 3] = np.count_nonzero(envel >= np.max(envel) * db3)
    pixels_6dB[i, 3] = np.count_nonzero(envel >= np.max(envel) * db6)
    pixels_9dB[i, 3] = np.count_nonzero(envel >= np.max(envel) * db9)
    max_pixel[i, 3] = np.max(envel)
    plt.imshow(envel[:,30,:], aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]],
               vmin=0, vmax=0.0185)
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('Hexagon/Circle scattering_angle=180')

    plt.subplot(2, 3, 4)
    envel = post_proc.envelope(ret_ret_12, 0)
    pixels_3dB[i, 4] = np.count_nonzero(envel >= np.max(envel) * db3)
    pixels_6dB[i, 4] = np.count_nonzero(envel >= np.max(envel) * db6)
    pixels_9dB[i, 4] = np.count_nonzero(envel >= np.max(envel) * db9)
    max_pixel[i, 4] = np.max(envel)
    plt.imshow(envel[:,30,:], aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]],
               vmin=0, vmax=0.0185)
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('Rectangle/Rectangle scattering_angle=12')

    plt.subplot(2, 3, 5)
    envel = post_proc.envelope(ret_circ_12, 0)
    pixels_3dB[i, 5] = np.count_nonzero(envel >= np.max(envel) * db3)
    pixels_6dB[i, 5] = np.count_nonzero(envel >= np.max(envel) * db6)
    pixels_9dB[i, 5] = np.count_nonzero(envel >= np.max(envel) * db9)
    max_pixel[i, 5] = np.max(envel)
    plt.imshow(envel[:,30,:], aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]],
               vmin=0, vmax=0.0185)
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('Rectangle/Circle scattering_angle=12')

    plt.subplot(2, 3, 6)
    envel = post_proc.envelope(hexag_circ_12, 0)
    pixels_3dB[i, 6] = np.count_nonzero(envel >= np.max(envel) * db3)
    pixels_6dB[i, 6] = np.count_nonzero(envel >= np.max(envel) * db6)
    pixels_9dB[i, 6] = np.count_nonzero(envel >= np.max(envel) * db9)
    max_pixel[i, 6] = np.max(envel)
    plt.imshow(envel[:,30,:], aspect='equal',
               extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]],
               vmin=0, vmax=0.0185)
    plt.xlabel('x[mm]')
    plt.ylabel('z[mm]')
    plt.title('Hexagon/Circle scattering_angle=12')

    plt.suptitle('Square side / Circle diameter = ' + rect_rect_size + 'mm')



    plt.show()



    print('Saving figure ' + i_config.__str__())
    plt.savefig('./summary/summary_'+i_config.__str__()+'.png')
    plt.close(fig)

    i += 1

plt.figure()
plt.title('-6 dB voxel count; scat_angle=180')
plt.plot(sizes, pixels_3dB[:,1], '-o')
plt.plot(sizes, pixels_3dB[:,2], '-x')
plt.plot(sizes, pixels_3dB[:,3], '-s')
minimum = np.min([pixels_3dB[:,1],pixels_3dB[:,2],pixels_3dB[:,3]])
plt.yscale('log')
plt.xlabel('element side or diameter')
plt.ylabel('(minimum='+minimum.__str__()+')')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

plt.figure()
plt.plot(sizes, pixels_3dB[:,4], '-o')
plt.plot(sizes, pixels_3dB[:,5], '-x')
plt.plot(sizes, pixels_3dB[:,6], '-s')
minimum = np.min([pixels_3dB[:,4],pixels_3dB[:,5],pixels_3dB[:,6]])
plt.yscale('log')
plt.xlabel('element side or diameter')
plt.ylabel('(minimum='+minimum.__str__()+')')
plt.title('-6 dB voxel count; scat_angle=12')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

plt.figure()
plt.plot(sizes, pixels_6dB[:,1], '-o')
plt.plot(sizes, pixels_6dB[:,2], '-x')
plt.plot(sizes, pixels_6dB[:,3], '-s')
minimum = np.min([pixels_6dB[:,1],pixels_6dB[:,2],pixels_6dB[:,3]])
plt.yscale('log')
plt.xlabel('element side or diameter')
plt.ylabel('(minimum='+minimum.__str__()+')')
plt.title('-12 dB voxel count; scat_angle=180')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

plt.figure()
plt.plot(sizes, pixels_6dB[:,4], '-o')
plt.plot(sizes, pixels_6dB[:,5], '-x')
plt.plot(sizes, pixels_6dB[:,6], '-s')
minimum = np.min([pixels_6dB[:,4],pixels_6dB[:,5],pixels_6dB[:,6]])
plt.yscale('log')
plt.xlabel('element side or diameter')
plt.ylabel('(minimum='+minimum.__str__()+')')
plt.title('-12 dB voxel count; scat_angle=12')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

plt.figure()
plt.plot(sizes, pixels_9dB[:,1], '-o')
plt.plot(sizes, pixels_9dB[:,2], '-x')
plt.plot(sizes, pixels_9dB[:,3], '-s')
minimum = np.min([pixels_9dB[:,1],pixels_9dB[:,2],pixels_9dB[:,3]])
plt.yscale('log')
plt.xlabel('element side or diameter')
plt.ylabel('(minimum='+minimum.__str__()+')')
plt.title('-18 dB voxel count; scat_angle=180')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

plt.figure()
plt.plot(sizes, pixels_9dB[:,4], '-o')
plt.plot(sizes, pixels_9dB[:,5], '-x')
plt.plot(sizes, pixels_9dB[:,6], '-s')
minimum = np.min([pixels_9dB[:,4],pixels_9dB[:,5],pixels_9dB[:,6]])
plt.yscale('log')
plt.xlabel('element side or diameter')
plt.ylabel('(minimum='+minimum.__str__()+')')
plt.title('-18 dB voxel count; scat_angle=12')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()


plt.figure()
plt.plot(sizes, max_pixel[:,1], '-o')
plt.plot(sizes, max_pixel[:,2], '-x')
plt.plot(sizes, max_pixel[:,3], '-s')
minimum = np.min([max_pixel[:,1],max_pixel[:,2],max_pixel[:,3]])
plt.yscale('log')
plt.xlabel('element side or diameter')
global_max_str = "{:.4f}".format(np.max(max_pixel[:, 1:4]))
plt.ylabel('global max: ' + global_max_str)
plt.title('Max amplitude (envelope); scat_angle=180')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

plt.figure()
plt.plot(sizes, max_pixel[:,4], '-o')
plt.plot(sizes, max_pixel[:,5], '-x')
plt.plot(sizes, max_pixel[:,6], '-s')
minimum = np.min([max_pixel[:,4],max_pixel[:,5],max_pixel[:,6]])
plt.yscale('log')
plt.xlabel('element side or diameter')
global_max_str = "{:.4f}".format(np.max(max_pixel[:, 4:7]))
plt.ylabel('global max: ' + global_max_str)
plt.title('Max amplitude (envelope); scat_angle=12')
plt.legend(['Rect/Rect', 'Rect/Circ', 'Hexag/Circ'])
plt.grid()

print('FINISHED')