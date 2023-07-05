from PyQt5.QtWebEngineWidgets import QWebEngineView
from framework import post_proc
from PyQt5 import QtCore, QtWidgets
import os
from framework.data_types import ImagingROI, ElementGeometry
import plotly.graph_objects as go
import numpy as np

import sys
import numpy as np


def show_isosurfaces(roi, img_result, min_db=1, max_db=9, contours=6):
    # PARAMETERS
    # roi: object of type framework.data_types.ImagingROI
    # img_result: 3D array resulting from imaging algorithm
    # min_db and max_db: minimum and maximum thresholds to be represented by isosurfaces
    # contours number of isosurfaces within the limits min_db and max_db (limits included)
    #
    # NOTES
    # - The isosurfaces thresholds are linearly spaced from min_db to max_db

    if min_db > max_db:
        min_db, max_db = max_db, min_db

    x_min = np.min(roi.get_coord()[:, 0])
    x_max = np.max(roi.get_coord()[:, 0])
    y_min = np.min(roi.get_coord()[:, 1])
    y_max = np.max(roi.get_coord()[:, 1])
    z_min = np.min(roi.get_coord()[:, 2])
    z_max = np.max(roi.get_coord()[:, 2])
    x, y, z = \
        np.mgrid[x_min:x_max:roi.w_len * 1j, y_min:y_max:roi.d_len * 1j, z_min:z_max:roi.h_len * 1j]

    img_result_env_db = np.abs(img_result)
    img_result_env_db = 10 * np.log10(np.abs(img_result_env_db.transpose((2, 1, 0))) + 1e-20)
    img_result_env_db = img_result_env_db - np.max(img_result_env_db)

    fig = go.Figure()
    fig.add_trace(go.Isosurface(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=img_result_env_db.flatten(),
        opacity=0.4,
        isomin=np.max(img_result_env_db) - max_db,
        isomax=np.max(img_result_env_db) - min_db,
        surface_count=contours,
        caps=dict(x_show=False, y_show=False)
    ))
    fig.add_trace(go.Mesh3d(
        x=[x_min, x_max, x_max, x_min],
        y=[y_min, y_min, y_max, y_max],
        z=[10, 10, 10, 10],
        opacity=0.4,
        color='gray'))
    fig.update_layout(scene=dict(zaxis=dict(nticks=4,
                                            range=[z_max, -1.],
                                            showspikes=False),
                                 xaxis=dict(showspikes=False),
                                 yaxis=dict(showspikes=False),
                                 hovermode=False))

    fig_view = QWebEngineView()
    html_filename = 'figura.html'
    fig.write_html(html_filename, config={'displayModeBar': False})
    html_path = os.path.join(os.getcwd(), html_filename)
    html_path = html_path.replace(os.sep, '/')
    fig_view.setUrl(QtCore.QUrl("file:///" + html_path))
    # fig_view.show()
    # fig_view.raise_()
    return fig_view


app = QtWidgets.QApplication(sys.argv)

data = np.load('C:/data/mirror/2021-01-19/Esferas/RetangularRetangular5mmCentro.var/2021-01-22-21-56-12/TFM pitch=1.5mm scatangle=None.npy', allow_pickle=True)
roi = data[1]
img_result = data[0]

fig_view = show_isosurfaces(roi, img_result, min_db=3, max_db=9, contours=3)