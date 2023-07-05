import numpy as np
import pyqtgraph.opengl as gl
from PyQt5 import QtWidgets
from pyqtgraph import parametertree
from scipy.spatial.distance import cdist

from framework.post_proc import normalize, envelope
from framework.utils import img_line
from framework.data_types import ImagingROI
from guiqt.Utils.ParameterRoot import ParameterRoot
from guiqt.Widgets import SurfaceEstimationWidgetDesign
from parameter_estimation.intsurf_estimation import profile_fadmm
from imaging.cumulative_tfm import cumulative_tfm_kernel


class InternalSurfaceWidget(QtWidgets.QWidget, SurfaceEstimationWidgetDesign.Ui_Form):
    def __init__(self, main_window):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.gl_widget.setBackgroundColor('w')

        self.main_window = main_window
        self.parameter_root = ParameterRoot()

        self.button_estimate.clicked.connect(self.estimate_button_pressed)
        self.estimated_surf = None

        roi_pars = [
            {'title': 'ROI parameters', 'readonly': True, 'name': 'roipar'},
            {'title': 'X Coordinate [mm]', 'type': 'float',
             'value': -20, 'readonly': False, 'name': 'x'},

            {'title': 'Y Coordinate [mm]', 'type': 'float',
             'value': 0, 'readonly': False, 'name': 'y'},

            {'title': 'Z Coordinate [mm]', 'type': 'float',
             'value': 0, 'readonly': False, 'name': 'z'},

            {'title': 'Height [mm]', 'type': 'float',
             'value': 20, 'readonly': False, 'name': 'h'},

            {'title': 'Points in Height', 'type': 'float',
             'value': 80, 'readonly': False, 'name': 'nh'},

            {'title': 'Width [mm]', 'type': 'float',
             'value': 40, 'readonly': False, 'name': 'w'},

            {'title': 'Points in Width', 'type': 'float',
             'value': 100, 'readonly': False, 'name': 'nw'},

            {'title': 'Data parameters', 'readonly': True, 'name': 'datpar'},
            {'title': 'Shots', 'type': 'ndarray',
             'value': np.array([0]), 'readonly': False, 'name': 'shots'},

            {'title': 'Estimation parameters', 'readonly': True, 'name': 'estpar'},
            {'title': 'Lambda', 'type': 'float',
             'value': 1, 'readonly': False, 'name': 'lambda'},

        ]

        self.parameter_root.addChildren(roi_pars)
        self.parameter_tree.addParameters(self.parameter_root, showTop=False)

    def estimate_button_pressed(self):
        d = {
            'ROI_h': self.parameter_root.getValues()['h'][0],
            'ROI_w': self.parameter_root.getValues()['w'][0],
            'ROI_nh': self.parameter_root.getValues()['nh'][0],
            'ROI_nw': self.parameter_root.getValues()['nw'][0],
            'ROI_x': self.parameter_root.getValues()['x'][0],
            'ROI_y': self.parameter_root.getValues()['y'][0],
            'ROI_z': self.parameter_root.getValues()['z'][0],
            'lamb': self.parameter_root.getValues()['lambda'][0],
            'sel_shots': self.parameter_root.getValues()['shots'][0],
        }
        if self.main_window.dados.inspection_params.type_insp == 'contact':
            self.main_window.run_in_thread(self.estimate, d, self.show_surface, show_overlay=True)
        else:
            try:
                if self.main_window.dados.surf.fitted:
                    self.main_window.run_in_thread(self.estimate, d, self.show_surface, show_overlay=True)
            except AttributeError:
                raise ValueError("Surface não inicializado")

    def estimate(self, ROI_h, ROI_w, ROI_nh, ROI_nw, ROI_x, ROI_y, ROI_z, lamb, sel_shots):
        corner_roi = np.array([ROI_x, ROI_y, ROI_z])[np.newaxis, :]
        roi = ImagingROI(corner_roi, ROI_h, ROI_nh, ROI_w, ROI_nw, 1, 1)
        key = cumulative_tfm_kernel(self.main_window.dados, roi, sel_shots=sel_shots)
        eroi = self.main_window.dados.imaging_results[key].roi
        y = normalize(envelope(self.main_window.dados.imaging_results[key].image))
        a = img_line(y)
        z = eroi.h_points[a[0]]

        zh, *args = profile_fadmm(a[1], z, lamb, rho=100, eta=0.999, itmax=500, max_iter_cg=1500)

        # zh = np.tile(zh, (20, 1)).T

        return -zh, eroi.w_points

    def show_surface(self, surf):
        [item._setView(None) for item in self.gl_widget.items]
        self.gl_widget.items = []
        self.gl_widget.update()
        xb = surf[1]
        zb = surf[0]
        xt = xb[:]
        if self.main_window.dados.inspection_params.type_insp == 'contact':
            zt = np.zeros_like(zb)
            ext_surf = gl.GLSurfacePlotItem(y=np.arange(20), x=xt, z=np.tile(zt, (20, 1)).T,
                                            shader='shaded', color=(0.7, 0.0, 0.0, 0.6))
        else:
            zt = -self.main_window.dados.surf.z_discr
            xt = self.main_window.dados.surf.x_discr
            # zt = np.tile(zt, (20, 1)).T
            ext_surf = gl.GLSurfacePlotItem(y=np.arange(20), x=xt, z=np.tile(zt, (20, 1)).T,
                                            shader='shaded', color=(0.7, 0.0, 0.0, 0.3))

        self.gl_widget.addItem(ext_surf)
        ext_surf.scale(0.01, 0.01, 0.01)
        # ext_surf.translate(-zb.shape[0] / 2, -zb.shape[1] / 2, 0, local=True)
        int_surf = gl.GLSurfacePlotItem(x=xb, y=np.arange(20), z=np.tile(zb, (20, 1)).T,
                                        shader='shaded', color=(0.0, 0.0, 0.7, 1))
        self.gl_widget.addItem(int_surf)
        int_surf.scale(0.01, 0.01, 0.01)
        xti = xb
        zti = np.interp(xb, xt, zt)
        dists = np.sqrt((xb[np.newaxis]-xti[:, np.newaxis])**2+(zb[np.newaxis].T-zti[:, np.newaxis])**2)
        mindepth = dists.min()
        self.depth.setText(f"{mindepth:.2f}")
        # int_surf.translate(-surf.shape[0] / 2, -surf.shape[1] / 2, 0, local=True)


