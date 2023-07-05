# -*- coding: utf-8 -*-
"""
Módulo ``ImagingTabWidget``
===========================

.. raw:: html

    <hr>

"""

import importlib
import os

import numpy as np
import pyqtgraph as pg
import plotly.graph_objects as go

from PyQt5 import QtCore, QtWidgets
from framework import post_proc

from guiqt.Widgets import ImagingTabDesign
from guiqt.Utils.ParameterRoot import ParameterRoot
from guiqt.Windows.ErrorWindow import ErrorWindow

from surface.surface import Surface, SurfaceType


class ImagingTabWidget(QtWidgets.QTabWidget, ImagingTabDesign.Ui_Form):
    """
    Classe utilziada pelo ``ImagingWidget``. Implementa uma aba para a execução dos algoritmos de imageamento.
    Possui uma seleção do algoritmo desejado, uma lista editável com os parâmetros do algoritmo, a imagem
    resultante do algoritmo e um *check box* para mostrar o envelope da imagem.
    """
    def __init__(self, centralwidget, alg_list, main_window, imaging, tab, index):
        super(self.__class__, self).__init__()
        self.setupUi(centralwidget)
        self.tab = tab
        self.main_window = main_window
        self.imaging = imaging
        if not self.main_window.has_data:
            self.frame_surf.setDisabled(True)
            self.frame_alg.setDisabled(True)
        else:
            self.frame_alg.setDisabled(False)
            self.frame_surf.setDisabled(self.main_window.dados.inspection_params.type_insp == 'contact')
        self.check_box_surf.setEnabled(False)

        self.param_root = ParameterRoot()

        self.img_proc = None
        self.img_pos_proc = None
        self.axis_limits = [0, 0, 0, 0]
        self.depth_limits = [0, 0]
        self.params = {}
        if self.main_window.dados.surf is None:
            self.main_window.dados.surf = Surface(self.main_window.dados)
        self.surface = None
        self.key = None
        self.index = index
        self.tab_name = None
        self.roi = None

        # conecta os sinais necessarios
        self.button_surf.clicked.connect(self.action_surf)
        self.box_alg.currentIndexChanged.connect(self.alg_changed)
        self.button_alg.clicked.connect(self.action_alg)
        self.check_box_envelope.stateChanged.connect(self.envelope_changed)
        self.spin_box_depth_2d.valueChanged.connect(self.envelope_changed)
        self.button_3d_draw.clicked.connect(self.draw_3d)
        self.check_box_surf.stateChanged.connect(self.surf_changed)
        self.radio_button_2d.clicked.connect(self.switch_to_2d)
        self.radio_button_3d.clicked.connect(self.switch_to_3d)

        self.plot_widget_2d.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plot_widget_2d.customContextMenuRequested.connect(self.show_pop_menu_img)
        self.plot_widget_2d.getPlotItem().setMenuEnabled(False)
        self.plot_widget_3d.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plot_widget_3d.customContextMenuRequested.connect(self.show_pop_menu_img)
        self.plot_widget_3d.getPlotItem().setMenuEnabled(False)

        for i in alg_list:
            self.box_alg.addItem(i)

        for i in SurfaceType:
            self.box_surf.addItem(i.name)
        # for i in range(len(alg_list)):
        #     self.box_alg.addItem(alg_list[i])

        self.img_dir_2d = pg.ImageView()
        self.img_dir_3d = pg.ImageView()
        self.plot_widget_2d.addItem(self.img_dir_2d.getImageItem())
        self.plot_widget_3d.addItem(self.img_dir_3d.getImageItem())
        self.plot_widget_2d.getPlotItem().setMenuEnabled(False)
        self.plot_widget_2d.getPlotItem().invertY()
        self.plot_widget_3d.getPlotItem().setMenuEnabled(False)
        self.plot_widget_3d.getPlotItem().invertY()

        self.splt_2d = pg.PlotDataItem(pen=pg.mkPen(width=2, color='r'))
        self.splt_3d = pg.PlotDataItem(pen=pg.mkPen(width=2, color='r'))
        self.plot_widget_2d.addItem(self.splt_2d)
        self.plot_widget_3d.addItem(self.splt_3d)

        self.plot_widget = self.plot_widget_2d

        self.spin_box_3d_min_dB.setValue(3)
        self.spin_box_3d_max_dB.setValue(9)
        self.spin_box_3d_contours.setMinimum(1)
        self.spin_box_3d_contours.setValue(3)

    def draw(self, img=None, rect=None):
        """ Desenha uma imagem no ``PlotWidget``. Caso não receba os limites da imagem, utiliza os valores salvos
        previamente.

        Parameters
        ----------
            img : `numpy.ndarray`
                Imagem a ser desenhada.

            rect : `PyQt5.QtCore.QRectF` ou None
                Limites dos eixos da imagem.
        """
        if img is None:
            img = self.img_proc
        img = np.real(img)
        if len(img.shape) == 2:
            img_dir = self.img_dir_2d
            image = img
            max = np.max(np.abs(image))
        else:
            img_dir = self.img_dir_3d
            image = img[:, self.spin_box_depth_2d.value(), :]
            max = np.max(np.abs(img))
        try:
            img_dir.getImageItem().setImage(post_proc.normalize(image.T, image_max=max, image_min=-max), levels=(0, 1))
        except Exception as e:
            ErrorWindow("Error while drawing image: " + e.args[0])
            return

        # corrige a escala de cores
        img_dir.getImageItem().setLookupTable(self.main_window.lut)
        if rect is None:
            rect = QtCore.QRectF(self.axis_limits[0], self.axis_limits[1],
                                 self.axis_limits[2] - self.axis_limits[0],
                                 self.axis_limits[3] - self.axis_limits[1])

        else:
            self.axis_limits = [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]

        img_dir.getImageItem().setRect(rect)

    def draw_3d(self):
        if len(self.img_proc.shape) != 3:
            return
        min_db = self.spin_box_3d_min_dB.value()
        max_db = self.spin_box_3d_max_dB.value()
        contours = self.spin_box_3d_contours.value()
        roi = self.roi

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

        img_result_env_db = post_proc.envelope(self.img_proc, 0)
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
        fig.update_layout(scene=dict(zaxis=dict(nticks=4,
                                                range=[z_max, z_min],
                                                showspikes=False),
                                     xaxis=dict(showspikes=False),
                                     yaxis=dict(showspikes=False),
                                     hovermode=False),
                          margin=dict(l=0, r=0, t=0, b=0))
        html_filename = 'figura.html'
        fig.write_html(html_filename, config={'displayModeBar': False})
        html_path = os.path.join(os.getcwd(), html_filename)
        url = QtCore.QUrl().fromLocalFile(html_path)
        self.web_engine_view.setUrl(url)

    def action_surf(self):
        """ Função chamada ao clicar no botao para executar o algoritmo de estimação de superfície externa.
         Prepara os dados necessários e executa o algoritmo na *thread* da janela principal.
        """
        try:
            d = {'surf_type': SurfaceType(self.box_surf.currentIndex()+1),
                 'shot': self.main_window.spin_box_shot.value(),
                 'sel_shots': self.param_root.get_parameters()['sel_shots']}
                 # 'roi': self.roi}
        except KeyError:
            d = {'surf_type': SurfaceType(self.box_surf.currentIndex()+1),
                 'shot': self.main_window.spin_box_shot.value(),
                 'sel_shots': self.main_window.spin_box_shot.value()}

        self.main_window.run_in_thread(self.main_window.dados.surf.fit, d, self.action_surf_finished, show_overlay=False)

    def action_surf_finished(self, *args):
        # print('Finished estimating surface')
        return

    def action_alg(self):
        """ Função chamada ao clicar no botao para executar o algoritmo. Prepara os dados necessários e executa o
        algoritmo na *thread* da janela principal.
        """
        # encontra o modulo do algoritmo selecionado
        alg_index = self.box_alg.currentIndex()
        str_modulo = self.box_alg.itemText(alg_index)
        modulo = importlib.import_module('.' + str_modulo, 'imaging')
        alg_kernel = getattr(modulo, str_modulo + '_kernel')

        # preenche um dicionario com os valores da arvore
        params = self.param_root.get_parameters()

        for key in params:
            self.params[key] = params[key]

        d = {'data_insp': self.main_window.dados, **params, 'roi': self.main_window.roi_framework,
             'sel_shot': self.main_window.spin_box_shot.value(), 'output_key': None}

        self.main_window.run_in_thread(alg_kernel, d, self.action_alg_finished, show_overlay=False)

    def action_alg_finished(self, key):
        """ Chamada ao finalizar o algoritmo de imageamento. Calcula os eixos e desenha a imagem na tela.

        Parameters
        ----------
            key : `numpy.int32`
                Chave gerada pelo algoritmo de imageamento.
        """
        if self.key is not None:
            self.main_window.dados.imaging_results.pop(self.key)
        self.key = key
        self.img_proc = self.main_window.dados.imaging_results[key].image
        if len(self.img_proc.shape) == 2:
            self.stacked_widget_imaging_tab.setCurrentIndex(0)
        elif len(self.img_proc.shape) == 3:
            self.stacked_widget_imaging_tab.setCurrentIndex(1)
            self.radio_button_2d.setChecked(True)
            self.stacked_widget_2d_3d.setCurrentIndex(0)

        self.img_pos_proc = None

        self.roi = self.main_window.dados.imaging_results[key].roi
        if len(self.img_proc.shape) == 2:
            xi = self.roi.coord_ref[0, 0]
            xf = xi + self.roi.width
            zi = self.roi.coord_ref[0, 2]
            zf = zi + self.roi.height

        else:
            xi = self.roi.coord_ref[0, 0]
            xf = xi + self.roi.width
            yi = self.roi.coord_ref[0, 1]
            yf = yi + self.roi.depth
            zi = self.roi.coord_ref[0, 2]
            zf = zi + self.roi.height
            self.depth_limits = [yi, yf]

            self.spin_box_depth_2d.setMaximum(self.img_proc.shape[1] - 1)

        self.axis_limits = [xi, zi, xf, zf]

        self.envelope_changed()
        self.draw_3d()

        self.tab.setTabText(self.index, self.main_window.dados.imaging_results[key].description)
        self.tab_name = self.main_window.dados.imaging_results[key].description

        try:
            self.surface = self.main_window.dados.imaging_results[key].surface
        except AttributeError:
            self.surface = None
        if self.surface is not None:
            self.check_box_surf.setEnabled(True)
            self.surf_changed()

        else:
            self.check_box_surf.setChecked(False)
            self.check_box_surf.setDisabled(True)

        # atualiza o combo box da pagina de analise
        if self.main_window.analysis_window is not None:
            self.main_window.analysis_window.update_tabs(self.imaging.tabs)


    def alg_changed(self):
        """ Chamada quando o usuário altera o algoritmo selecionado. Preenche a árvore de parâmetros com os
        parâmetros do novo algoritmo.
        """
        # encontra o modulo do algoritmo selecionado
        alg_index = self.box_alg.currentIndex()
        str_modulo = self.box_alg.itemText(alg_index)
        try:
            modulo = importlib.import_module('.' + str_modulo, 'imaging')
            alg_param = getattr(modulo, str_modulo + '_params')
            getattr(modulo, str_modulo + '_kernel')  # apenas para garantir que a interface ira encontrar o kernel do
            # modulo
        except AttributeError:
            ErrorWindow("Could not load algorithm and/or parameters")
            self.box_alg.setCurrentIndex(1 if self.box_alg.currentIndex() == 0 else 0)
            return

        dict_param = alg_param()

        self.tree_alg.clear()
        self.param_root = ParameterRoot()
        for key, value in dict_param.items():
            if key != 'roi':
                type_val = type(value).__name__
                # se o parametro ja foi utilizado anteriormente, apenas copia
                if key == 'description':
                    self.param_root.addChild({'name': key, 'type': 'str', 'value': str_modulo})
                elif key in self.params:
                    self.param_root.addChild({'name': key, 'type': type_val, 'value': self.params[key],
                                              'decimals': 12})
                # se o valor padrao passado for None, irá considerar o tipo como float
                elif key == 'output_key' or key == 'sel_shot':
                    pass
                elif type_val == 'NoneType':
                    self.param_root.addChild({'name': key, 'type': 'float', 'value': -1, 'decimals': 12})
                elif key == 'c':
                    self.param_root.addChild({'name': key, 'type': 'float',
                                              'value': self.main_window.dados.specimen_params.cl,
                                              'default': self.main_window.dados.specimen_params.cl,
                                              'decimals': 12})
                elif key == 'angles':
                    if self.main_window.dados.inspection_params.type_capt == "PWI":
                        ang = self.main_window.dados.inspection_params.angles
                        readonly = True
                    else:
                        ang = value
                        readonly = False
                    self.param_root.addChild({'name': key, 'type': type_val, 'value': ang, 'decimals': 12,
                                              'readonly': readonly})
                else:
                    self.param_root.addChild({'name': key, 'type': type_val, 'value': value, 'decimals': 12})

        self.tree_alg.addParameters(self.param_root, showTop=False)

    def envelope_changed(self):
        """ Desenha a imagem com ou sem envelope, dependendo do valor do `check box`. Caso seja a primeira vez
        que irá desenhar o envelope da imagem, salva o resultado para não precisar recalcular.
        """
        if self.img_proc is None:
            return
        if self.check_box_envelope.isChecked():
            if self.img_pos_proc is None:
                self.img_pos_proc = post_proc.envelope(self.img_proc, 0)
            self.draw(self.img_pos_proc)
        else:
            self.draw(self.img_proc)

    def surf_changed(self):
        """ Desenha ou esconde uma linha com a superfície na imagem.
        """
        if self.check_box_surf.isChecked():
            idx = np.where(np.logical_and(self.surface[0] >= self.axis_limits[0],
                                          self.surface[0] <= self.axis_limits[2]))[0]
            self.splt_2d.setData(self.surface[0][idx[1:-1]], self.surface[1][idx[1:-1]])
            self.splt_3d.setData(self.surface[0][idx[1:-1]], self.surface[1][idx[1:-1]])
        else:
            self.splt_2d.clear()
            self.splt_3d.clear()
            if self.check_box_envelope.isChecked():
                self.draw(self.img_pos_proc)
            else:
                self.draw(self.img_proc)

    def change_lut(self, lut):
        """ Muda o mapa de cores da imagem.

        Parameters
        ----------
            lut : `String`
                Nome do mapa de cor escolhido.
        """
        try:
            self.plot_widget_2d.getPlotItem().items[0].setLookupTable(lut)
            self.plot_widget_3d.getPlotItem().items[0].setLookupTable(lut)
        except (AttributeError, IndexError):
            pass

    def show_pop_menu_img(self, point):
        """ Mostra o `pop menu` da imagem.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.
        """
        action = self.imaging.pop_menu_img.exec_(self.plot_widget.mapToGlobal(point))
        self.treat_action(action, 'i')

    def treat_action(self, action, pos):
        """ Trata as ações dos menus de contexto da imagem.

        Parameters
        ----------
            action : `String`
                Ação escolhida pelo usuário.

            pos : `String`
                Identifica o widget que chamou a função.
        """
        if action is None:
            return

        elif action.text() == 'Save':
            if pos == 'i':  # imagem da esquerda
                img = self.plot_widget
            else:
                return
            self.main_window.save_image(img)

        else:  # muda colormap
            self.main_window.change_lut(action.text())

    def disable(self, b):
        """ Desativa os *widgets* de interação com o usuário durante a execução de algum algoritmo na *thread* da
        janela principal.
        Parameters
        ----------
            b : `bool`
                Flag para desativar ou ativar os widgets.
        """
        if b or self.main_window.dados.inspection_params.type_insp == 'contact':
            self.frame_surf.setDisabled(True)
        else:
            self.frame_surf.setDisabled(False)
        self.frame_alg.setDisabled(b)
        # self.button_alg.setDisabled(b)
        # self.check_box_envelope.setDisabled(b)
        # self.box_alg.setDisabled(b)
        # self.tree_alg.setDisabled(b)
        if not b and self.surface is not None:
            self.frame_surf.setEnabled(True)
            self.check_box_surf.setEnabled(True)
        else:
            self.check_box_surf.setChecked(False)
            self.check_box_surf.setDisabled(True)

    def disable_surf(self, b):
        """ Desativa as opções de surface.
        Parameters
        ----------
            b : `bool`
                Flag para desativar ou ativar os widgets.
        """
        self.frame_surf.setDisabled(b)

    def switch_to_2d(self):
        self.stacked_widget_2d_3d.setCurrentIndex(0)
        self.plot_widget = self.plot_widget_2d

    def switch_to_3d(self):
        self.stacked_widget_2d_3d.setCurrentIndex(1)
        self.plot_widget = self.plot_widget_3d
