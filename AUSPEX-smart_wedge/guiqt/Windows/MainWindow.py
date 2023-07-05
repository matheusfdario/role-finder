# -*- coding: utf-8 -*-
"""
Módulo ``MainWindow``
=====================

Implementa a janela principal da interface gráfica.

.. raw:: html

    <hr>

"""

import os
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMenu, QAction, QFileDialog, QDialog, QMainWindow, QInputDialog
from matplotlib import cm, colors  # utiliza os colormaps do matplotlib
from pyqtgraph import exporters, parametertree

import imaging
from framework import file_m2k, file_civa, file_omniscan, post_proc, pre_proc
from framework.data_types import DataInsp, ImagingROI
from guiqt.Windows import MainWindowDesign

from guiqt.Utils import Cmaps
from guiqt.Utils.Overlay import Overlay
from guiqt.Utils.ParameterRoot import ParameterRoot
from guiqt.Utils.Thread import Thread
from guiqt.Utils.ArrayParameter import ArrayParameter
from guiqt.Utils.Console import Console

from guiqt.Windows.ErrorWindow import ErrorWindow
from guiqt.Windows.PreProcWindow import PreProcWindow
from guiqt.Windows.AnalysisWindow import AnalysisWindow
from guiqt.Windows.ShotsSelectionWindow import ShotsSelectionWindow

from guiqt.Widgets.ImagingWidget import ImagingWidget
from guiqt.Widgets.EstimationWidget import EstimationWidget
from guiqt.Widgets.SurfaceEstimationWidget import InternalSurfaceWidget


class MainWindow(QMainWindow, MainWindowDesign.Ui_MainWindow):
    """
    Janela principal da interface, com um ``QTabWidget`` para instanciar as abas de cada funcionalidade do *framework*.
    Possui a visualização dos dados brutos e de A-scan, com seleção de disparo e de ROI.

    A definição da ROI é feita na janela principal, e os outros *widgets* com as funcionalidades do framework podem
    acessar o atributo **roi_framework**, que é sempre atualizado com a ROI mostrada na tela.

    Utiliza os colormaps presentes no ``matplotlib`` e o mapa de cores padrão do CIVA, que também é o mapa padrão da
    interface. Para trocar, basta clicar com o botão direito em uma imagem e selecionar a opção *colormaps*. Os mapas
    disponíveis serão listados, e todas as imagens serão atualizadas com o mesmo mapa. Todos os *widgets* da aba deverão
    chamar a função da janela principal quando houver a requisição para a mudança de *colormap*. Cada *widget* deve
    criar uma função para alterar seus mapas de cor.

    Possui uma função para salvar ``PlotWidgets``. Permite salvar como um vetor do ``Numpy`` (.npy) ou como uma imagem
    (.png e .svg). Os *widgets* da aba podem chamar essa função, passando o ``PlotWidget`` que será salvo, e a janela
    principal faz a interação com o usuário para selecionar o nome e tipo do arquivo que será salvo. Cada *widget* deve
    criar sua própria forma de selecionar o ``PlotWidget`` a ser salvo. Para salvar os gráficos da janela principal,
    basta clicar com o botão direito na imagem desejada e selecionar **Save**.
    """
    finished_sig = QtCore.pyqtSignal()

    def __init__(self):
        """Construtor da classe.
        """
        super(self.__class__, self).__init__()
        self.setupUi(self)
        np.set_printoptions(precision=1)
        # conecta os sinais necessarios
        self.action_open_dir.triggered.connect(self.open_dir)
        self.action_open_file.triggered.connect(self.open_file)
        self.action_save_result.triggered.connect(self.save_pickle)
        self.action_open_result.triggered.connect(self.open_pickle)
        self.action_change_salvo.triggered.connect(self.change_salvo)
        self.action_open_pre_proc.triggered.connect(self.open_pre_proc)
        self.action_open_analysis.triggered.connect(self.open_analysis)
        self.action_open_console.triggered.connect(self.open_console)
        self.action_open_documentation.triggered.connect(self.open_documentation)
        self.action_open_cl_estimator.triggered.connect(self.cl_estimation_clicked)
        self.action_open_internal_surface_estimator.triggered.connect(self.int_surf_estimation_clicked)
        self.spin_box_channel.valueChanged.connect(self.spinbox_changed)
        self.spin_box_sequence.valueChanged.connect(self.spinbox_changed)
        self.spin_box_shot.valueChanged.connect(self.spinbox_changed)

        # cria menu de contexto das imagens
        self.pop_menu_img = QMenu(self)

        # cria opcao para salvar imagem
        menu_save = QAction("Save", self.pop_menu_img)
        self.pop_menu_img.addAction(menu_save)

        # lista colormaps disponiveis
        menu_cmap = QMenu("Colormaps", self.pop_menu_img)
        self.colormaps = [m for m in plt.colormaps() if not m.endswith("_r")]
        self.colormaps.append('civa')
        for cmap in self.colormaps:
            action = QAction(cmap, self)
            action.setCheckable(True)
            if cmap == 'civa':
                action.setChecked(True)
            else:
                action.setChecked(False)
            menu_cmap.addAction(action)

        self.pop_menu_img.addMenu(menu_cmap)

        # opcao de habilitar ROI na imagem
        self.action = QAction('Show ROI', self)
        self.action.setCheckable(True)
        self.action.setChecked(False)
        self.show_roi = False
        self.pop_menu_img.addAction(self.action)

        # mantem como padrao o colormap do civa
        self.cmap = 'civa'
        self.lut = Cmaps.civa

        # cria o colormap do civa no matplotlib - para salvar svg
        self.civa_cmap = colors.ListedColormap(self.lut/255)

        # configura para mostrar os menus de contexto
        self.plot_widget_esq.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plot_widget_esq.customContextMenuRequested.connect(self.show_pop_menu_img_esq)
        self.plot_widget_esq.getPlotItem().setMenuEnabled(False)

        self.plot_widget_ascan.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plot_widget_ascan.customContextMenuRequested.connect(self.show_pop_menu_img_ascan)
        self.plot_widget_ascan.getPlotItem().setMenuEnabled(False)

        # ROI do framework e da interface
        self.roi_qt = pg.ROI([0, 0])
        self.roi_framework = ImagingROI()

        # eixos das imagens [xi, zi, xf, zf]
        self.img_rect_esq = [0, 0, 0, 0]

        # maximo dos ascans carregados
        self.ascan_max = 0

        # raiz da arvore de parametros
        self.parameters_root = ParameterRoot()

        # registra novos parametros
        pg.parametertree.parameterTypes.registerParameterType('ndarray', ArrayParameter)

        # o DataInsp é mantido pela classe da janela principal
        self.dados = DataInsp()

        # Caso o arquivo seja multisalvo, mantem uma lista com todos os DataInsp
        self.data_insp_list = []

        # caminho para o arquivo aberto
        self.file = ''

        # Flags para controle interno
        self.drawing = False
        self.roi_changing = False
        self.readonly_params = False
        self.has_data = False

        # Cria thread e overlay para ser mostrando enquanto thread executa
        self.overlay = Overlay(self.centralwidget)
        self.thread = Thread(self)
        # funcao para ser chamada ao finalizar a execucao da thread
        self.function_to_call = None

        # conecta o sinal emitido pela thread ao finalizar
        self.finished_sig.connect(self.thread_finished)

        # cria as abas
        # imageamento

        self.tabs = {}

        self.add_tab(ImagingWidget(self), 'Imaging')
        self.analysis_window = None

        # cria o console
        namespace = {'data_insp': self.dados, 'np': np, 'imaging': imaging, 'file_civa': file_civa,
                     'file_m2k': file_m2k, 'file_omniscan': file_omniscan, 'pre_proc': pre_proc, 'post_proc': post_proc,
                     '_mw': self, 'plt': plt, 'roi': self.roi_framework}
        self.console = Console(**namespace)
        self.console.setWindowTitle('AUSPEX: console')

        # altera o stdout padrao para o console
        # sys.stdout = self.console.kernel.stdout

    def open_dir(self):
        """ Função chamada para selecionar e abrir um diretório (.CIVA/.m2k)
        Cria um ``FileDialog`` padrão do sistema operacional a partir do ``Qt`` para selecionar o diretório, e
        seleciona a função do *framework* correta a ser usada.
        """
        # cria um file dialog para selecionar o diretorio do arquivo
        file_dir = pg.FileDialog()
        file = file_dir.getExistingDirectory()

        # cria a estrutura com os dados do ensaio
        if file[-4:] == ".m2k":
            sel_shots_window = ShotsSelectionWindow()
            sel_shots = sel_shots_window.sel_shot
            if sel_shots == -1:
                return
            d = {'filename': file, 'type_insp': "immersion", 'water_path': 0.0, 'freq_transd': 5.0, 'bw_transd': 0.6,
                 'tp_transd': "gaussian", "sel_shots": sel_shots}
            func = file_m2k.read
            self.readonly_params = False

        elif file[-5:] == ".civa":
            sel_shots_window = ShotsSelectionWindow()
            sel_shots = sel_shots_window.sel_shot
            if sel_shots == -1:
                return
            d = {'filename': file, 'sel_shots': sel_shots}
            func = file_civa.read
            self.readonly_params = False
        else:
            if file:
                ErrorWindow("File type not supported")
            return

        self.file = file

        self.has_data = True
        self.run_in_thread(func, d, self.finished_open_dir)

    def finished_open_dir(self, data_insp):
        """ Função chamada após abrir um arquivo. Um novo ``DataInsp`` será salvo, as imagens e parâmetros serão
        atualizados, assim como ROI e *spinboxes*.
        Ao fim da função, chama a função do ``ImagingWidget`` e a do ``EstimationWidget`` para atualizar o que for
        necessário.

        Parameters
        ----------
            data_insp : :class:`framework.DataInsp`
                Objeto com os dados carregados pelo *framework*.
        """
        if isinstance(data_insp, list):
            self.data_insp_list = data_insp
            data_insp_names = []
            for i in range(len(data_insp)):
                data_insp_names.append(str(i) + ': ' + data_insp[i].dataset_name)

            name, ok = QInputDialog.getItem(self, "Multisalvo selection", "Salvo:", data_insp_names, 0, False)

            if not ok:
                return
            idx = int(name.split(':')[0])

            data_insp = data_insp[idx]

            self.action_change_salvo.setEnabled(True)

        else:
            self.data_insp_list = []

        # mantem os imaging_results antigos
        for key in self.dados.imaging_results:
            data_insp.imaging_results[key] = self.dados.imaging_results[key]
        self.dados = data_insp

        # atualiza o data_insp do console
        self.console.kernel_manager.kernel.shell.push({'data_insp': self.dados})

        # cria uma roi
        zi = 0
        zf = self.dados.time_grid.shape[0]

        if self.dados.probe_params.type_probe == 'linear':
            xi = self.dados.probe_params.elem_center[0][0]
            xf = self.dados.probe_params.elem_center[-1][0]

        elif self.dados.probe_params.type_probe == 'matricial' or self.dados.probe_params.type_probe == 'generic':
            xi = self.dados.probe_params.elem_center[:, 0].min()
            xf = self.dados.probe_params.elem_center[:, 0].max()

        else:  # self.dados.probe_params.type_probe is 'mono':
            xi = self.dados.inspection_params.step_points[0][0]
            xf = self.dados.inspection_params.step_points[-1][0]

        if self.dados.inspection_params.type_insp == 'immersion':
            # mantem o transdutor em (0, 0, 0)
            zf -= zi
            zi -= zi

        self.img_rect_esq = [xi, zi, xf, zf]

        self.roi_changing = True
        self.roi_qt = pg.ROI([xi, zi], [xf - xi, zf - zi], maxBounds=QtCore.QRectF(xi - (xf-xi)*0.5, zi - (zf-zi)*0.5,
                                                                                   (xf - xi)*2, (zf - zi)*2))
        self.roi_qt.addScaleHandle([1, 0.5], [0, 0.5])
        self.roi_qt.addScaleHandle([0.5, 1], [0.5, 0])
        self.roi_qt.addScaleHandle([0, 0.5], [1, 0.5])
        self.roi_qt.addScaleHandle([0.5, 0], [0.5, 1])
        self.plot_widget_esq.getPlotItem().addItem(self.roi_qt)
        self.roi_qt.sigRegionChangeFinished.connect(self.roi_changed_img)

        # coloca os valores necessarios tabelas da interface
        self.parameters_root.clearChildren()
        self.parametertree.clear()
        self.parametertree.setHeaderHidden(True)

        # nome do arquivo aberto
        file_par = pg.parametertree.Parameter(name='File path', expanded=False)
        file_par.addChild({'name': self.file})

        # Para facilitar a atualizacao dos dados do DataInsp quando um desses parametros eh alterado, eh passado para
        # 'title' o nome a ser mostrado para o usuario e para 'name' o caminho desse parametro no DataInsp.
        # parametros da inspeção
        insp_pars = [
            {'title': 'Inspection Type', 'name': 'inspection_params.type_insp', 'type': 'list',
             'values': {"Immersion": 'immersion', "Contact": "contact"},
             'value': self.dados.inspection_params.type_insp, 'readonly': self.readonly_params},

            {'title': 'Excitation', 'name': 'inspection_params.type_capt', 'type': 'str',
             'value': self.dados.inspection_params.type_capt, 'readonly': True},

            {'title': 'Origin [mm]', 'name': 'inspection_params.point_origin', 'type': 'str',
             'value': f"{self.dados.inspection_params.point_origin}", 'readonly': True},

            {'title': 'Water Path [mm]', 'name': 'inspection_params.water_path', 'type': 'float',
             'value': self.dados.inspection_params.water_path if self.dados.inspection_params.water_path is not None
             else 0, 'readonly': self.readonly_params},

            {'title': 'Couplant L-Speed [m/s]', 'name': 'inspection_params.coupling_cl', 'type': 'float',
             'value': self.dados.inspection_params.coupling_cl, 'readonly': self.readonly_params,
             'decimals': 6},

            {'title': 'Sample Frequency [MHz]', 'name': 'inspection_params.sample_freq', 'type': 'float',
             'value': self.dados.inspection_params.sample_freq, 'readonly': self.readonly_params},

            {'title': 'Gate start [us]', 'name': 'inspection_params.gate_start', 'type': 'float',
             'value': self.dados.inspection_params.gate_start, 'readonly': self.readonly_params},

            {'title': 'Nb. Samples', 'name': 'inspection_params.gate_samples', 'type': 'float',
             'value': self.dados.inspection_params.gate_samples, 'readonly': self.readonly_params,
             'decimals': 6},

            {'title': 'Hardware Gain [dB]', 'name': 'inspection_params.gain_hw', 'type': 'float',
             'value': self.dados.inspection_params.gain_hw, 'readonly': True,
             'decimals': 6},

            {'title': 'Digital Gain [dB]', 'name': 'inspection_params.gain_sw', 'type': 'float',
             'value': self.dados.inspection_params.gain_sw, 'readonly': True,
             'decimals': 6},
        ]
        insp_par = pg.parametertree.Parameter(name='Inspection Parameters', children=insp_pars)
        # liga os sinais de mudanca vindos do ``Parameter`` para atualizar o ``parameter_tree``
        for child in insp_par:
            child.sigValueChanged.connect(self.value_changed_tree)

        # parametros do probe
        elem_dim = np.asarray(self.dados.probe_params.elem_dim) if \
            hasattr(self.dados.probe_params.elem_dim, "__len__") else self.dados.probe_params.elem_dim
        elem_dim_type = 'ndarray' if hasattr(elem_dim, "__len__") else 'float'
        probe_pars = [
            {'title': 'Probe Type', 'name': 'self.dados.probe_params.type_probe', 'type': 'str',
             'value': self.dados.probe_params.type_probe, 'readonly': self.readonly_params},

            {'title': 'Element Dimension [mm]', 'name': 'self.dados.probe_params.elem_dim', 'type': elem_dim_type,
             'value': elem_dim,
             'readonly': self.readonly_params},

            {'title': 'Central Frequency [MHz]', 'name': 'self.dados.probe_params.contral_freq', 'type': 'float',
             'value': self.dados.probe_params.central_freq, 'readonly': self.readonly_params},

            {'title': 'Pulse Bandwidth [-6dB]', 'name': 'self.dados.probe_params.bw', 'type': 'float',
             'value': self.dados.probe_params.bw, 'readonly': self.readonly_params}
        ]

        if self.dados.probe_params.type_probe == 'linear':
            probe_pars.append({'title': 'Nb. Elements', 'name': 'self.dados.probe_params.num_elem', 'type': 'int',
                               'value': self.dados.probe_params.num_elem, 'readonly': self.readonly_params})

            probe_pars.append({'title': 'Pitch [mm]', 'name': 'self.dados.probe_params.pitch', 'type': 'float',
                               'value': self.dados.probe_params.pitch, 'readonly': self.readonly_params})

        probe_par = pg.parametertree.Parameter(name='Probe Parameters', children=probe_pars)
        for child in probe_par:
            child.sigValueChanged.connect(self.value_changed_tree)

        # parametros do objeto de inspeção
        spec_pars = [
            {'title': 'L-Speed in material [m/s]', 'name': 'self.dados.specimen_params.cl', 'type': 'float',
             'value': self.dados.specimen_params.cl, 'readonly': self.readonly_params, 'decimals': 6},
            {'title': 'T-Speed in material [m/s]', 'name': 'self.dados.specimen_params.cs', 'type': 'float',
             'value': self.dados.specimen_params.cs, 'readonly': self.readonly_params, 'decimals': 6},
            {'title': 'Surface Roughness [mm]', 'name': 'self.dados.specimen_params.roughness', 'type': 'float',
             'value': self.dados.specimen_params.roughness, 'readonly': self.readonly_params, 'decimals': 6},

        ]
        spec_par = pg.parametertree.Parameter(name='Specimen Parameters', children=spec_pars)
        for child in spec_par:
            child.sigValueChanged.connect(self.value_changed_tree)

        dt = self.dados.time_grid[1][0] - self.dados.time_grid[0][0]
        zi = dt * zi * self.dados.specimen_params.cl * 10e-4 * 0.5 \
            + self.dados.probe_params.elem_center[0, 2]
        zf = dt * zf * self.dados.specimen_params.cl * 10e-4 * 0.5 \
            + self.dados.probe_params.elem_center[0, 2]

        # parametros da ROI
        roi_pars = [
            {'name': 'X Coordinate [mm]', 'type': 'float', 'value': xi},
            {'name': 'Y Coordinate [mm]', 'type': 'float', 'value': 0},
            {'name': 'Z Coordinate [mm]', 'type': 'float', 'value': zi},
            {'name': 'Height [mm]', 'type': 'float', 'value': zf-zi, 'limits': (0, sys.maxsize)},
            {'name': 'Pixels in height', 'type': 'float', 'value': self.roi_framework.h_len,
             'limits': (2, sys.maxsize)},
            {'name': 'Width [mm]', 'type': 'float', 'value': self.roi_qt.boundingRect().width(),
             'limits': (0, sys.maxsize)},
            {'name': 'Pixels in width', 'type': 'float', 'value': self.roi_framework.w_len, 'limits': (2, sys.maxsize)},
            {'name': 'Depth [mm]', 'type': 'float', 'value': 10, 'limits': (0, sys.maxsize)},
            {'name': 'Pixels in depth', 'type': 'float', 'value': self.roi_framework.d_len, 'limits': (1, sys.maxsize)},
        ]
        roi_par = pg.parametertree.Parameter(name="ROI", children=roi_pars)

        self.parametertree.addParameters(file_par)
        self.parameters_root.addChild(file_par)
        self.parametertree.addParameters(insp_par)
        self.parameters_root.addChild(insp_par)
        self.parametertree.addParameters(probe_par)
        self.parameters_root.addChild(probe_par)
        self.parametertree.addParameters(spec_par)
        self.parameters_root.addChild(spec_par)
        self.parametertree.addParameters(roi_par)
        self.parameters_root.addChild(roi_par)

        self.roi_changing = False

        # limita os spinboxes
        self.spin_box_sequence.setMaximum(self.dados.ascan_data.shape[1] - 1)
        self.spin_box_channel.setMaximum(self.dados.ascan_data.shape[2] - 1)
        self.spin_box_shot.setMaximum(self.dados.ascan_data.shape[3] - 1)

        # liga os sinais de mudanca vindos do ``Parameter`` da ROI com a funcao para atualizar a ROI
        for child in roi_par:
            child.sigValueChanged.connect(self.roi_changed_tree)

        self.ascan_max = np.max(np.abs(self.dados.ascan_data))
        self.draw_bscan()

        self.roi_changed_img()

        # informa os widgets da atualizacao
        for tab_name in self.tabs:
            try:
                self.tabs[tab_name].file_opened()
            except AttributeError:
                pass

    # Abrir .opd
    def open_file(self):
        """ Semelhante à ``open_dir()``, abre um arquivo que não é um diretório."""
        # cria um file dialog para selecionar o diretorio do arquivo
        # file_dir = pg.FileDialog()
        file = QFileDialog.getOpenFileName(self, "Select File", "", "*.opd")[0]

        # cria a estrutura com os dados do ensaio
        if file[-4:] == ".opd":
            sel_shots_window = ShotsSelectionWindow()
            sel_shots = sel_shots_window.sel_shot
            if sel_shots == -1:
                return
            d = {'filename': file, 'sel_shots': sel_shots, 'freq': 5.0, 'bw': 0.6,
                 'pulse_type': "gaussian"}
            func = file_omniscan.read
            self.readonly_params = False
        else:
            if file:
                ErrorWindow("File type not supported")
            return

        self.file = file

        self.has_data = True

        self.run_in_thread(func, d, self.finished_open_dir)

    def open_pickle(self):
        """ Função temporária para leitura de dados criados com a biblioteca ``pickle``.
        Espera uma classe do tipo ``ImagingResults``.
        """
        filepath = QFileDialog.getOpenFileName(self, 'Open File')
        if filepath[0] == '':
            return
        file = open(filepath[0], 'rb')
        results = pickle.load(file)
        for key in results:
            aux_key = key

            while True:
                if aux_key not in self.dados.imaging_results:
                    break
                ii32 = np.iinfo(np.int32)
                aux_key = np.random.randint(low=ii32.min, high=ii32.max, dtype=np.int32)

            self.dados.imaging_results[aux_key] = results[key]
        file.close()

        self.tabs['Imaging'].file_opened(pickle=results)

    def save_pickle(self):
        """ Salva os resultados dos algoritmos de imageamento em um arquivo através do módulo ``pickle``.
        Salva o arquivo com extensão **.AUSPEX**.
        """
        if self.dados.imaging_results.__len__() == 0:
            ErrorWindow('No results to save')
            return
        filepath = QFileDialog.getSaveFileName(self, 'Save File')
        if filepath[0] == '':
            return
        if filepath[0].endswith('.AUSPEX'):
            path = filepath[0][:-7]
        else:
            path = filepath[0]
        outfile = open(path + '.AUSPEX', 'wb')
        pickle.dump(self.dados.imaging_results, outfile)
        outfile.close()

    def draw_bscan(self, img=None):
        """ Desenha um B-scan no ``PlotWidget`` esquerdo. Pode também desenhar uma imagem qualquer.
        Caso não seja passada uma imagem, irá desenhar um B-scan dos dados presentes no ``DataInsp`` carregado, e os
        eixos serão calculados automaticamente, na escala de milímetros no eixo horizontal, e amostras no vertical.
        O *slice* é escolhido através dos `spinboxes` presentes na tela.

        Parameters
        ----------
            img : :class:`numpy.ndarray` ou None
                Imagem a ser desenhada. Também pode ser ´None´ para desenhar um B-scan dos dados carregados.
        """
        self.drawing = True

        img_esq = pg.ImageView()  # cria um imageview
        emissor = self.spin_box_sequence.value()
        shot = self.spin_box_shot.value()

        # coloca a imagem do arquivo no imageview
        if img is None:
            max = self.ascan_max
            if self.dados.inspection_params.type_capt == "sweep":
                # se for sweep, mostra todos os passos

                img = post_proc.normalize(np.real(self.dados.ascan_data[:, 0, 0, :]), image_max=max, image_min=-max)

            elif self.dados.inspection_params.type_capt == "FMC":
                # mostra o passo selecionado
                img = post_proc.normalize(np.real(self.dados.ascan_data[:, emissor, :, shot]), image_max=max, image_min=-max)
            elif self.dados.inspection_params.type_capt == "PWI":
                # mostra o passo e angulo selecionado
                img = post_proc.normalize(np.real(self.dados.ascan_data[:, emissor, :, shot]), image_max=max, image_min=-max)

            elif self.dados.inspection_params.type_capt == "Unisequential":
                img = post_proc.normalize(np.real(self.dados.ascan_data[:, emissor, :, shot]), image_max=max, image_min=-max)

            elif self.dados.inspection_params.type_capt == "FMC_sum":
                img = post_proc.normalize(np.real(self.dados.ascan_data[:, :, emissor, shot]), image_max=max,
                                          image_min=-max)

            else:
                ErrorWindow("Only possible for the following excitation types: sweep, FMC, PWI or Unisequential")
                return
        if np.isnan(img.min()) or np.isinf(img.min()):
            self.plot_widget_esq.getPlotItem().clear()
            return

        img_esq.setImage(img.T, levels=(0, 1))

        img_esq.getImageItem().setLookupTable(self.lut)

        # mostra a imagem
        self.plot_widget_esq.getPlotItem().clear()
        self.plot_widget_esq.addItem(img_esq.getImageItem())

        # inverte a direção do eixo y
        img_esq.getImageItem().getViewBox().invertY()

        # calcula os eixos
        if img is None:
            # se passou a imagem, nao calcula os eixos
            pass
        else:
            limits = QtCore.QRectF(self.img_rect_esq[0], 0,
                                   self.img_rect_esq[2] - self.img_rect_esq[0],
                                   img.shape[0])
            img_esq.getImageItem().setRect(limits)

        # redesenha a roi
        if self.show_roi:
            self.plot_widget_esq.getPlotItem().addItem(self.roi_qt)

        # centraliza a imagem
        self.plot_widget_esq.getPlotItem().autoRange()

        self.drawing = False

    def draw_ascan(self):
        """ Desenha o A-scan determinado pelos `spinboxes` na tela.
        """
        channel = self.spin_box_channel.value()
        sequence = self.spin_box_sequence.value()
        shot = self.spin_box_shot.value()

        ascan = np.real(self.dados.ascan_data[:, sequence, channel, shot])

        self.plot_widget_ascan.getPlotItem().clear()
        self.plot_widget_ascan.addItem(pg.PlotDataItem(ascan))

    def spinbox_changed(self):
        """ Chamada quando algum `spinbox` é alterado. Desenha o novo A-scan e B-scan.
        """
        # redesenha a imagem
        self.draw_bscan()

        # redesenha a-scan
        self.draw_ascan()

    def roi_changed_tree(self):
        """ Função chamada quando algum parâmetro referente à ROI na árvore de parâmetros é alterado.
        Altera os valores da ROI visual na interface. Possui uma flag para evitar *loops* com ``roi_changed_img()``.
        """
        if self.roi_changing:
            return

        self.roi_changing = True
        roi_parameters = self.parameters_root.get_roi_parameters()
        px = roi_parameters["X Coordinate [mm]"]
        py = roi_parameters["Y Coordinate [mm]"]
        pz = roi_parameters["Z Coordinate [mm]"]

        h = roi_parameters["Height [mm]"]
        d = roi_parameters["Depth [mm]"]
        w = roi_parameters["Width [mm]"]

        dimx = roi_parameters["Pixels in width"]
        dimy = roi_parameters["Pixels in depth"]
        dimz = roi_parameters["Pixels in height"]

        # checa os limites da imagem
        if px + w > self.img_rect_esq[2] + w*0.5:
            w = self.img_rect_esq[2] - px + w*0.5

        if pz + h > self.img_rect_esq[3] + h*0.5:
            h = self.img_rect_esq[3] - pz + h*0.5

        if px < self.img_rect_esq[0] - w*0.5:
            px = self.img_rect_esq[0] - w*0.5

        if pz < self.img_rect_esq[1] - h*0.5:
            pz = self.img_rect_esq[1] - h*0.5

        cl = float(self.parameters_root.get_parameters("Specimen Parameters")['self.dados.specimen_params.cl'])

        dt = self.dados.time_grid[1][0] - self.dados.time_grid[0][0]

        self.roi_qt.setPos((px, pz/(dt * cl * 10e-4 * 0.5)))

        self.roi_qt.setSize([w, h/(dt * cl * 10e-4 * 0.5)])

        coord_ref = np.zeros((1, 3))
        coord_ref[0, 0] = px
        coord_ref[0, 1] = py
        coord_ref[0, 2] = pz

        self.roi_framework = ImagingROI(coord_ref, height=h, width=w, depth=d, h_len=dimz, w_len=dimx, d_len=dimy)
        self.console.kernel_manager.kernel.shell.push({'roi': self.roi_framework})

        self.roi_changing = False

    def value_changed_tree(self, param):
        """ Chamada quando algum valor na árvore de parâmetros é alterado. Altera o valor no ``DataInsp`` carregado.
        Parameters
        ----------
            param : :class:`pg.parametertree.Parameter`
                Parâmetro alterado.
        """
        child = param.name().split('.')[-1]
        parent = param.name().split('.')[-2]
        setattr(getattr(self.dados, parent), child, param.value())
        if child == 'type_insp':
            self.tabs['Imaging'].disable_surf(not(param.value() == 'immersion'))

    def roi_changed_img(self):
        """ Função chamada quando a ROI na imagem é alterada.
        Altera os valores da ROI na árvore de parâmetros. Possui uma flag para evitar *loops* com
        ``roi_changed_tree()``.
        """
        if self.roi_changing:
            return

        self.roi_changing = True
        # atualiza a roi do ``DataInsp``
        [xi, zi, xf, zf] = self.roi_qt.parentBounds().getCoords()

        cl = self.dados.specimen_params.cl

        dt = self.dados.time_grid[1][0] - self.dados.time_grid[0][0]

        zi = dt * zi * cl * 10e-4 * 0.5 \
            + self.dados.probe_params.elem_center[0, 2]
        zf = dt * zf * cl * 10e-4 * 0.5 \
            + self.dados.probe_params.elem_center[0, 2]

        coord_ref = np.zeros((1, 3))
        coord_ref[0, 0] = xi
        coord_ref[0, 1] = self.parameters_root.get_roi_parameters()["Y Coordinate [mm]"]
        coord_ref[0, 2] = zi

        parameters_old = self.parameters_root.get_roi_parameters()
        p_h = parameters_old["Pixels in height"]
        p_w = parameters_old["Pixels in width"]

        self.roi_framework = ImagingROI(coord_ref, height=zf-zi, width=xf-xi, h_len=p_h, w_len=p_w)
        self.console.kernel_manager.kernel.shell.push({'roi': self.roi_framework})

        # atualiza os valores na tabela de parametros
        self.parameters_root.set_roi_parameters(coord_ref, zf-zi, xf-xi)

        self.roi_changing = False

        self.draw_ascan()

    def save_image(self, plotwidget):
        """ Salva um ``plotwidget`` utilizando ``PyQtGraph``. Caso a extensão do arquivo seja .svg, cria uma imagem com o
        ``Matplotlib`` para salvar, pois existe um bug na funcionalidade do ``PyQtGraph``.

        Parameters
        ----------
            plotwidget : :class:`pg.plotwidget`
                ``Plotwidget`` que será salvo.
        """
        # formatos disponiveis para salvar
        types = 'Numpy array (*.npy);;PNG (*.png);;SVG (*.svg)'

        if plotwidget is None:
            return

        try:
            plotwidget.getPlotItem().items[0]
        except IndexError:
            return

        # cria janela para escolher diretorio e nome do arquivo
        file_dialog = QFileDialog()
        file_dialog.setOption(QFileDialog.DontResolveSymlinks)
        name = file_dialog.getSaveFileName(self, 'Save File', '', types)

        if name[0] is '':  # cancelou
            return

        type = name[1].split('*')[1].split(')')[0]
        name = name[0].replace(type, '')

        if type == '.npy':
            try:
                img = plotwidget.getPlotItem().items[0].image
            except AttributeError:
                img = plotwidget.getPlotItem().items[0].getData()
            np.save(name, img)

        elif type == '.png':
            # esconde a roi, caso esteja na imagem
            if plotwidget.getPlotItem().items.__len__() > 1:
                plotwidget.getPlotItem().items[1].hide()
            exporter = pg.exporters.ImageExporter(plotwidget.scene())
            exporter.export(name + type)
            if plotwidget.getPlotItem().items.__len__() > 1:
                plotwidget.getPlotItem().items[1].show()

        elif type == '.svg':
            # usa matplotlib para salvar, pois o exportador do pyqtgraph possui problemas com os eixos da imagem
            xi = self.img_rect_esq[0]
            zi = self.img_rect_esq[1]
            xf = self.img_rect_esq[2]
            zf = self.img_rect_esq[3]

            # passa o eixo Y invertido
            if self.cmap == 'civa':
                plt.imshow(plotwidget.getPlotItem().items[0].image.T, extent=[xi, xf, zf, zi], cmap=self.civa_cmap,
                           vmin=0, vmax=1)
            else:
                plt.imshow(plotwidget.getPlotItem().items[0].image.T, extent=[xi, xf, zf, zi], cmap=self.cmap)
            plt.savefig(name + type)

    def change_lut(self, lut):
        """ Função chamada quando o usuário altera o *colormap* utilizado. O parâmetro `lut` é do tipo ``String``, com o
        nome do *colormap*.

        Parameters
        ----------
            lut : `String`
                Nome do mapa de cor escolhido.
        """
        self.cmap = lut
        if lut == 'civa':
            lut = Cmaps.civa
        else:
            cmap = cm.get_cmap(lut)
            cmap._init()
            lut = [[row[i] * 255 for i in range(3)] for row in cmap._lut[:256, 0:3]]
        self.lut = lut
        try:
            self.plot_widget_esq.getPlotItem().items[0].setLookupTable(lut)
        except (AttributeError, IndexError):
            return

        for tab_name in self.tabs:
            try:
                self.tabs[tab_name].change_lut(lut)
            except AttributeError:
                pass

    def show_pop_menu_img_esq(self, point):
        """ Mostra o `pop menu` da imagem esquerda.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.
        """
        action = self.pop_menu_img.exec_(self.plot_widget_esq.mapToGlobal(point))
        self.treat_action(action, 'l')

    def show_pop_menu_img_ascan(self, point):
        """ Mostra o `pop menu` do A-scan.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.
        """
        action = self.pop_menu_img.exec_(self.plot_widget_ascan.mapToGlobal(point))
        self.treat_action(action, 'a')

    def treat_action(self, action, pos):
        """ Função chamada quando o usuário seleciona alguma opção do menu de contexto.

        Parameters
        ----------
            action : `String`
                Ação escolhida pelo usuário.

            pos : `String`
                Identifica o widget que chamou a função.
        """
        # trata as ações dos menus de contexto das imagens
        if action is None:
            return

        elif action.text() == 'Save':
            if pos == 'l':  # imagem da esquerda
                img = self.plot_widget_esq
            elif pos == 'a':  # ascan
                img = self.plot_widget_ascan
            else:
                return
            self.save_image(img)

        elif action.text() == 'Show ROI':
            self.show_roi = action.isChecked()
            self.draw_bscan()

        else:  # muda colormap
            idx = self.colormaps.index(self.cmap)
            self.pop_menu_img.actions()[1].menu().actions()[idx].setChecked(False)
            self.change_lut(action.text())

    def open_pre_proc(self):
        """ Abre a janela de pre-processamento.
        """
        dialog = QDialog()
        PreProcWindow(dialog, self)
        self.draw_bscan()
        self.draw_ascan()

    def add_tab(self, widget, name):
        # estimacao de velocidade
        self.tabs[name] = widget
        idx = self.main_tab.addTab(widget, name)
        self.main_tab.setCurrentIndex(idx)

    def remove_tab(self, name):
        w = self.tabs.pop(name)
        self.main_tab.removeTab(self.main_tab.indexOf(w))

    def cl_estimation_clicked(self):
        if 'L-Speed estimation' in self.tabs:
            self.action_open_cl_estimator.setChecked(False)
            self.remove_tab('L-Speed estimation')
        else:
            self.action_open_cl_estimator.setChecked(True)
            self.add_tab(EstimationWidget(self), 'L-Speed estimation')

    def int_surf_estimation_clicked(self):
        if 'Internal surface' in self.tabs:
            self.action_open_internal_surface_estimator.setChecked(False)
            self.remove_tab('Internal surface')
        else:
            self.action_open_internal_surface_estimator.setChecked(True)
            self.add_tab(InternalSurfaceWidget(self), 'Internal surface')


    def run_in_thread(self, function_to_run, params_dict, function_to_call, show_overlay=True):
        """ Função padrão para executar uma função em *thread*. Por padrão, mostra um *overlay*, impedindo a interação
        do usuário com a interface, e fazendo com que a interface não pare de responder durante o processamento.

        Parameters
        ----------
            function_to_run : `method`
                Função a ser executada na *thread*.
            params_dict : `dict`
                Dicionário com os parâmetros da função.
            function_to_call : `method`
                Função chamada após a execução da thread.
            show_overlay : `bool`
                Flag para mostrar o *overlay* da interface.
        """
        # function_to_run e a funcao que sera rodada na thread, e function_to_call a funcao chamada apos a thread
        # terminar
        if show_overlay:
            self.action_open_dir.setDisabled(True)
            self.action_open_file.setDisabled(True)
            self.action_open_result.setDisabled(True)
            self.action_save_result.setDisabled(True)
            self.action_open_pre_proc.setDisabled(True)
            self.action_change_salvo.setDisabled(True)
            self.overlay.show()

        self.parametertree.setDisabled(True)

        for tab_name in self.tabs:
            try:
                self.tabs[tab_name].disable(True)
            except AttributeError:
                pass

        self.spin_box_shot.setDisabled(True)

        self.thread.set_dict(params_dict)
        self.thread.set_func(function_to_run)
        self.function_to_call = function_to_call
        self.thread.start()

    @pyqtSlot()
    def thread_finished(self):
        """ Função chamada ao terminar a execução da thread. Irá chamar a função passada como parâmetro para
        ``run_in_thread()`` com os valores obtidos pela função executada na thread. Caso a função levante uma exeção,
        uma janela de erro será mostrada.
        """
        self.action_open_dir.setEnabled(True)
        self.action_open_file.setEnabled(True)
        self.action_open_result.setEnabled(True)
        self.action_save_result.setEnabled(True)

        self.parametertree.setDisabled(False)

        if self.has_data:
            self.action_open_pre_proc.setEnabled(True)

        if len(self.data_insp_list) > 0:
            self.action_change_salvo.setEnabled(True)

        for tab_name in self.tabs:
            try:
                self.tabs[tab_name].disable(False)
            except AttributeError:
                pass

        self.spin_box_shot.setDisabled(False)

        self.overlay.hide()
        if self.thread.exception:
            if type(self.thread.result) == FileNotFoundError:
                self.has_data = False
                self.action_open_pre_proc.setDisabled(True)
            ErrorWindow(str(self.thread.result))
            return
        try:
            self.function_to_call(self.thread.result)
        except Exception as e:
            ErrorWindow(str(e))

    def open_analysis(self):
        """ Abre a janela de análise.
        """
        tabs = self.tabs['Imaging'].tabs
        self.analysis_window = AnalysisWindow(self, tabs)

    def open_console(self):
        """ Abre o console do IPython.
        As seguintes referências são salvas com os seguintes nomes:
        * DataInsp carregado na interface: data_insp
        * ROI atual na interface: roi
        * Módulo imaging: imaging
        * Módulo file_civa: file_civa
        * Módulo file_m2k: file_m2k
        * Módulo file_omniscan: file_omniscan
        * Módulo pre_proc: pre_proc
        * Módulo post_proc: post_proc
        * Módulo numpy: np
        * Módulo matplotlib.pyploy: plt
        * Widget da janela principal: _mw

        Tanto a roi quanto o data_insp são sincronizados com a interface gráfica, mas a interface não atualiza a
        visualização automaticamente quando um atributo é alterado.
        """
        self.console.show()

    @staticmethod
    def open_documentation():
        """ Abre a documentação em HTML no navegador padrão. Mostra uma mensagem de erro caso não encontre o índice no
        caminho padrão.
        """
        if getattr(sys, 'frozen', False):
            application_path = os.path.join('Auspex_Toolbox', sys._MEIPASS)
        else:
            application_path = os.getcwd()
        filename = os.path.join(application_path, 'documentation', '_build', 'html', 'index.html')
        if not os.path.isfile(filename):
            ErrorWindow("Documentation not found. If you are running the source code, build the html documentation and "
                        "put it in the following folder: {project_root}/documentation/'_build/html/ and start the GUI from the "
                        "script 'gui_run'.")
            return
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])

    def change_salvo(self):
        self.finished_open_dir(self.data_insp_list)
