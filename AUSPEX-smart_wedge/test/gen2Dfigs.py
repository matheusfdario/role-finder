import matplotlib.pyplot as plt
import numpy as np
from framework.data_types import ImagingROI
from framework import post_proc
import os
from datetime import datetime


def the_batch(pathroot, timestampnpy, doi):
    xdczerototal = -1

    path = pathroot + '/' + timestampnpy
    now = 'depth' + "{:.2f}".format(doi) + '_' + \
          str(datetime.now())[:-7].replace(':', '-').replace(' ', '-')
    os.mkdir(path + '/' + now)

    # Counts the number of .npy files as loads the filenames
    # to a list (called, guess what, filenames)
    filenames = list()
    dir_res = os.listdir(path)
    for elem in dir_res:
        if elem[-4:] == '.npy' and elem[:3] == 'TFM':
            filenames.append(elem)
    N = len(filenames)
    max_pixel = np.zeros(N)

    pitch_list = np.load(path + '/pitch_list.npy')

    corner_roi = np.array([-15.0, -15, 15.0])[np.newaxis, :]
    roi = ImagingROI(corner_roi, height=20.0, width=30.0, depth=30.0, h_len=40, w_len=60, d_len=61)

    # Nearest neighbor
    i_depth = np.argmin(np.abs(roi.d_points - doi))
    print('i_depth=' + str(i_depth))

    for i_config in range(N):
        tfm = np.load(path + '/' + filenames[i_config])

        envel = np.abs(tfm)

        fig = plt.figure()
        plt.imshow(envel[:, i_depth, :], aspect='equal',
                   extent=[roi.w_points[0], roi.w_points[-1], roi.h_points[-1], roi.h_points[0]])
        title = 'depth=' + "{:.2f}".format(doi) + 'mm' + \
                'size=' + "{:.2f}".format(pitch_list[i_config] - .1)
        plt.show()
        plt.xlabel('x[mm]')
        plt.ylabel('z[mm]')
        plt.title(title)
        plt.savefig(path + '/' + now + '/' + title + '.png')
        plt.close(fig)

    print('FINISHED')


the_batch('D:/cantos/RetangularRetangularCanto.var', '2021-01-17-11-59-44', 5.0)
