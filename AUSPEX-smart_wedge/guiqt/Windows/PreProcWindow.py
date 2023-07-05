# -*- coding: utf-8 -*-
"""
Módulo ``PreProcWindow``
========================

Implementa uma janela com funcionalidades de pré-processamento dos dados.

.. raw:: html

    <hr>

"""

import inspect

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from framework import file_m2k, file_civa, file_omniscan, post_proc, pre_proc

from guiqt.Windows import PreProcWindowDesign

from guiqt.Windows.ErrorWindow import ErrorWindow

from guiqt.Utils.ParameterRoot import ParameterRoot


class PreProcWindow(PreProcWindowDesign.Ui_pre_proc_dialog):
    """ Classe responsável por abrir uma janela para aplicar algoritmos de pré-processamento nos dados carregados pela
    janela principal. Os algoritmos são automaticamente reconhecidos, desde que estejam no arquivo
    ``framework/pre_proc.py``. É necessário que eles possuam ao menos dois parâmetros: ``data_insp`` e ``shots``, sendo
    o primeiro uma instância da classe ``DataInsp`` e o segundo um `numpy.ndarray` com os índices dos disparos em que o
    algoritmo será aplicado.
    """

    def __init__(self, dialog, main_window):
        """ Construtor da classe.

        Parameters
        ----------
            dialog : :class:`PyQt5.QtWidgets.QDialog`
                Janela de diálogo.

            main_window :class:`guiqt.gui.MainWindow`
                Janela principal.

        """
        self.setupUi(dialog)
        self.dialog = dialog
        dialog.setModal(True)

        self.main_window = main_window

        # encontra os algoritmos no modulo ``pre_proc``
        algs = [x[0] for x in inspect.getmembers(pre_proc, inspect.isfunction)]
        for i in range(len(algs)):
            self.combo_box_alg.addItem(algs[i])

        # cria a raiz da arvore de parametros
        self.parameters_root = ParameterRoot()

        # limita as spin boxes
        self.spin_box_sequence.setRange(0, self.main_window.dados.ascan_data.shape[1] - 1)
        self.spin_box_channel.setRange(0, self.main_window.dados.ascan_data.shape[2] - 1)

        # conecta os sinais
        self.combo_box_alg.currentIndexChanged.connect(self.alg_changed)
        self.button_apply.clicked.connect(self.visualize)
        self.button_save.clicked.connect(self.save)
        self.button_reset.clicked.connect(self.reset)
        self.button_resetall.clicked.connect(self.reset_all)
        self.spin_box_channel.valueChanged.connect(self.redraw)
        self.spin_box_sequence.valueChanged.connect(self.redraw)
        self.spin_box_shot.valueChanged.connect(self.redraw)

        # remove os menus de contexto
        self.plot_widget_ascan.setMenuEnabled(False)
        self.plot_widget_bscan.setMenuEnabled(False)

        self.alg_changed()

        try:
            self.draw_ascan(self.main_window.dados.ascan_data[:, 0, 0, self.main_window.spin_box_shot.value()])
            self.draw_bscan(self.main_window.dados.ascan_data[:, 0, :, self.main_window.spin_box_shot.value()])
        except Exception:  # a exceçao retornada nao e especifica
            return

        self.shot_pos = 0
        self.last_result = self.main_window.dados.ascan_data[:, :, :, :]

        shape = self.last_result.shape

        self.spin_box_sequence.setRange(0, shape[1] - 1)
        self.spin_box_channel.setRange(0, shape[2] - 1)
        self.spin_box_shot.setRange(0, shape[3] - 1)

        # remove botao '?'
        dialog.setWindowFlags(dialog.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)

        dialog.exec_()

    def draw_ascan(self, data):
        """ Desenha o A-scan do *slicing* selecionado.

        Parameters
        ----------
            data : :class:`numpy.ndarray`
                A-scan a ser desenhado.
        """
        self.plot_widget_ascan.getPlotItem().clear()
        self.plot_widget_ascan.addItem(pg.PlotDataItem(data))

    def draw_bscan(self, img):
        """ Desenha o B-scan com os dados presentes no ``DataInsp`` carregado.

        Parameters
        ----------
            img : :class:`numpy.ndarray`
                B-scan a ser desenhado.
        """
        img_bscan = pg.ImageView()  # cria um imageview

        # coloca a imagem no imageview
        max = np.max(np.abs(img))
        img_bscan.setImage(post_proc.normalize(img.T, image_max=max, image_min=-max), levels=(0, 1))
        img_bscan.getImageItem().setLookupTable(self.main_window.lut)

        # mostra a imagem
        self.plot_widget_bscan.getPlotItem().clear()
        self.plot_widget_bscan.addItem(img_bscan.getImageItem())

        # inverte a direção do eixo y
        img_bscan.getImageItem().getViewBox().invertY()

        # calcula os eixos
        if img is not None:
            # se passou a imagem, nao calcula os eixos
            pass
        else:
            limits = QtCore.QRectF(self.main_window.img_rect_esq[0], self.main_window.img_rect_esq[1],
                                   self.main_window.img_rect_esq[2] - self.main_window.img_rect_esq[0],
                                   self.main_window.img_rect_esq[3] - self.main_window.img_rect_esq[1])
            img_bscan.getImageItem().setRect(limits)

        # centraliza a imagem
        self.plot_widget_bscan.getPlotItem().autoRange()

    def alg_changed(self):
        """ Encontra os parâmetros do algoritmo selecionado. Assume que parâmetros com valor padrão ``None`` são
        considerados do tipo ``float``.
        """
        alg_index = self.combo_box_alg.currentIndex()
        func_str = self.combo_box_alg.itemText(alg_index)
        func = getattr(pre_proc, func_str)
        func_params = inspect.signature(func)
        params = [key for key in func_params.parameters.keys()]
        defaults = [func_params.parameters[key].default for key in params]

        self.parametertree.clear()
        self.parameters_root = ParameterRoot()
        # TODO: Usar ScalableGroup para adicionar os argumentos opcionais.
        for i in range(len(params)):
            if i == 0:
                continue  # o primeiro sempre é data_insp?
            if defaults[i] is inspect._empty:
                continue
            type_val = type(defaults[i]).__name__
            if type_val == 'NoneType':
                self.parameters_root.addChild({'name': params[i], 'type': 'float', 'value': 0, 'decimals': 12})
            elif params[i] == 'shots':
                self.parameters_root.addChild({'name': params[i], 'type': 'ndarray', 'value': defaults[i], 'limits':
                                               (0, self.main_window.dados.ascan_data.shape[3] - 1)})
            elif type_val == 'ndarray':
                self.parameters_root.addChild({'name': params[i], 'type': 'ndarray', 'value': defaults[i]})
            else:
                self.parameters_root.addChild({'name': params[i], 'type': type_val, 'value': defaults[i],
                                               'decimals': 12})

        self.parametertree.addParameters(self.parameters_root)

    def apply_alg(self):
        """ Executa o algoritmo selecionado.
        """
        alg_index = self.combo_box_alg.currentIndex()
        func_str = self.combo_box_alg.itemText(alg_index)
        func = getattr(pre_proc, func_str)
        try:
            self.shot_pos = self.parameters_root.get_parameters()['shots'].astype(int)
        except KeyError:
            self.shot_pos = int(self.parameters_root.get_parameters()['shot'])

        self.last_result = np.copy(self.main_window.dados.ascan_data[:, :, :, self.shot_pos], order='F')
        try:
            out = func(self.main_window.dados, **self.parameters_root.get_parameters())

            self.spin_box_sequence.setRange(0, out.shape[1] - 1)
            self.spin_box_channel.setRange(0, out.shape[2] - 1)
            self.spin_box_shot.setRange(0, out.shape[3] - 1)
            self.main_window.spin_box_sequence.setMaximum(out.shape[1] - 1)
            self.main_window.spin_box_channel.setMaximum(out.shape[2] - 1)
            self.main_window.spin_box_shot.setMaximum(out.shape[3] - 1)
            self.main_window.ascan_max = np.max(np.abs(out))
            return out
        except Exception as e:
            ErrorWindow("Error during preprocessing: " + e.args[0])
        return None

    def visualize(self):
        """ Aplica o algoritmo selecionado. O resultado deverá ser salvo pelo algoritmo.
        """
        out = self.apply_alg()
        if out is None:
            return
        seq = self.spin_box_sequence.value()
        chan = self.spin_box_channel.value()
        shot = self.spin_box_shot.value()
        self.draw_bscan(np.real(self.main_window.dados.ascan_data[:, seq, :, shot]))
        self.draw_ascan(np.real(self.main_window.dados.ascan_data[:, seq, chan, shot]))

    def save(self):
        """ Chamado quando o botão para salvar é clicado. Como o algoritmo deve salvar o resultado, a janela irá apenas
        fechar.
        """
        # Apenas fecha a janela
        self.dialog.close()

    def reset(self):
        """ Remove o ultimo processamento feito.
        """
        if self.last_result.shape.__len__() == 3:
            self.main_window.dados.ascan_data[:, :, :, self.shot_pos] = self.last_result[:, :, :]
        else:
            self.main_window.dados.ascan_data = self.last_result
        self.redraw()

    def reset_all(self):
        """ Recarrega os A-scan, abrindo o arquivo novamente.
        """
        if self.main_window.file[-4:] == ".m2k":
            d = {'filename': self.main_window.file, 'type_insp': "immersion", 'water_path': 0.0, 'freq_transd': 5.0,
                 'bw_transd': 0.6, 'tp_transd': "gaussian"}
            func = file_m2k.read
            self.main_window.readonly_params = False

        elif self.main_window.file[-5:] == ".civa":
            d = {'filename': self.main_window.file, 'sel_shots': None}
            func = file_civa.read
            self.main_window.readonly_params = True
        elif self.main_window.file[-4:] == ".opd":
            d = {'filename': self.main_window.file, 'sel_shots': 0, 'freq': 5.0, 'bw': 0.6,
                 'pulse_type': "gaussian"}
            func = file_omniscan.read
            self.main_window.readonly_params = False
        else:
            if self.main_window.file:
                ErrorWindow("Could not find file")
            return
        self.main_window.run_in_thread(func, d, self.reset_all_finished)

    def reset_all_finished(self, data_insp):
        self.main_window.finished_open_dir(data_insp)
        self.last_result = self.main_window.dados.ascan_data
        self.redraw()

    def redraw(self):
        """ Desenha novamente o A-scan e B-scan quando um *spin box* é alterado.
        """
        seq = self.spin_box_sequence.value()
        chan = self.spin_box_channel.value()
        shot = self.spin_box_shot.value()
        self.draw_bscan(np.real(self.main_window.dados.ascan_data[:, seq, :, shot]))
        self.draw_ascan(np.real(self.main_window.dados.ascan_data[:, seq, chan, shot]))
