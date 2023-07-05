import matplotlib.pyplot as plt
import numpy as np
from framework.data_types import ImagingROI
from framework import post_proc
import os


def subimg(roi: ImagingROI, img: np.ndarray,
           ref: np.ndarray, side: float):
    x_where = np.where(np.all(
        [roi.w_points >= ref[0] - side/2,
         roi.w_points <= ref[0] + side/2], 0))
    x_where = x_where[0]
    x_min = x_where[0]
    x_max = int(x_where[-1] + 1)

    y_where = np.where(np.all(
        [roi.d_points >= ref[1] - side / 2,
         roi.d_points <= ref[1] + side / 2], 0))
    y_where = y_where[0]
    y_min = y_where[0]
    y_max = int(y_where[-1] + 1)

    z_where = np.where(np.all(
        [roi.h_points >= ref[2] - side / 2,
         roi.h_points <= ref[2] + side / 2], 0))
    z_where = z_where[0]
    z_min = z_where[0]
    z_max = int(z_where[-1] + 1)

    print([x_min, x_max, y_min, y_max, z_min, z_max])

    return img[z_min:z_max, y_min:y_max, x_min:x_max]


def the_batch(pathroot, timestamp, ratio=False,
              voxel0=[0.,0.,25.], voxel1=[5.,5.,25.],
              sidetol=2.5):
    xdczerototal = -1

    db_scale = np.array([-3, -6, -9, -12, -15, -18])
    lin_scale = 10**(db_scale/20)

    path = pathroot + '/' + timestamp
    # Counts the number of .npy files as loads the filenames
    # to a list (called, guess what, filenames)
    filenames = list()
    dir_res = os.listdir(path)
    for elem in dir_res:
        if elem[-4:] == '.npy' and elem[:3] == 'TFM':
            filenames.append(elem)
    N = len(filenames)
    corner_roi = np.array([-15.0, -15, 15.0])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=61)
    max_pixel = np.zeros(N)
    ratios = np.zeros(N)

    pitch_list = np.load(path + '/pitch_list.npy')

    pixels_threshold = np.zeros((len(db_scale), N))
    images = list()

    for i_config in range(N):
        tfm = np.load(path + '/' + filenames[i_config], allow_pickle=True)[0]

        envel = np.abs(tfm)
        for i_gain in range(len(db_scale)):
            pixels_threshold[i_gain, i_config] = \
                np.count_nonzero(envel >= np.max(envel) * lin_scale[i_gain])
        max_pixel[i_config] = np.max(envel)

        if ratio is True:
            plt.figure()
            subimg0 = subimg(roi, envel, voxel0, sidetol)
            plt.subplot(121)
            plt.imshow(subimg0[:, 3, :])
            subimg1 = subimg(roi, envel, voxel1, sidetol)
            plt.subplot(122)
            plt.imshow(subimg1[:, 3, :])
            ratios[i_config] = subimg1.max() / subimg0.max()

    if ratio is not True:
        for i_gain in range(len(db_scale)):
            plt.figure()
            plt.title(str(db_scale[i_gain]) + ' dB voxel count')
            plt.plot(pitch_list - .1, pixels_threshold[i_gain, :], '-o')
            #plt.axis(np.array(plt.axis()) * [1, 0, 1, 1] + [0, 1.8, 0, 0])
            minimum = np.min(pixels_threshold[i_gain, :])
            plt.yscale('log')
            plt.xlabel('element side')
            plt.ylabel('(minimum=' + minimum.__str__() + ')')
            plt.grid()


    plt.figure()
    plt.plot(pitch_list - .1, max_pixel, '-o')
    print(max_pixel)
    #plt.axis(np.array(plt.axis()) * [1, 0, 1, 1] + [0, 1.8, 0, 0])
    minimum = np.min(max_pixel)
    plt.yscale('log')
    plt.xlabel('element side')
    global_max_str = "{:.4f}".format(np.max(max_pixel))
    plt.ylabel('global max: ' + global_max_str)
    plt.title('Max amplitude (envelope)')
    plt.grid()

    if ratio is True:
        plt.figure()
        plt.plot(pitch_list - .1, ratios, '-o')
        #plt.yscale('log')
        plt.xlabel('element side')
        global_max_str = "{:.4f}".format(np.max(ratios))
        plt.ylabel('max: ' + global_max_str)
        plt.title('Amplitude ratio')
        plt.grid()

    print('FINISHED')

    return max_pixel

the_batch('C:/data/RetangularRetangular1.5mmCentro.var', '2021-03-10-23-24-18')

#the_batch('C:/data/mirror/2021-01-19/Pontos/RetangularRetangularLadoRatio.var', '2021-01-28-00-30-31', True, [0., 0., 25.], [5., 0., 25.], sidetol=2.5)
#the_batch('C:/data/mirror/2021-01-19/Pontos/RetangularRetangularCantoRatio.var', '2021-01-27-21-24-06', True, [0., 0., 25.], [5., 5., 25.], sidetol=2.5)
#centro = the_batch('C:/data/mirror/2021-01-19/Pontos/RetangularRetangularCentro.var', '2021-01-27-20-40-44')
#lado = the_batch('C:/data/mirror/2021-01-19/Pontos/RetangularRetangularLado.var', '2021-01-14-19-22-10')

#the_batch('C:/data/esferas/2021-01-19-21-04-00.var', '2021-01-17-12-30-37', True, [0., 0., 25.], [5., 5., 25.], sidetol=2.5)
#lado = the_batch('D:/cantos/RetangularRetangularLado.var', '2021-01-17-13-00-36')
#centro = the_batch('D:/cantos/RetangularRetangularCentro.var', '2021-01-18-19-00-38')
#the_batch('D:/cantos/RetangularRetangularLadoRatio.var', '2021-01-17-13-28-43')

