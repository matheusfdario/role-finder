# -*- coding: utf-8 -*-
"""
Módulo ``EstimationWidget``
===========================

.. raw:: html

    <hr>

"""

import collections
import importlib
import os

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMenu, QAction

import imaging
from framework import post_proc
from guiqt.Widgets import EstimationWidgetDesign
from parameter_estimation import cl_estimators


class EstimationWidget(QtWidgets.QWidget, EstimationWidgetDesign.Ui_estimation):
    """
    Instancia um *widget* criado através do ``Designer`` que contem a aba relacionada aos algoritmos de estimação de
    velocidade longitudinal na peça do *framework*. Permite selecionar a métrica e o algoritmo de imageamento desejados,
    além da grade para fazer a busca.

    Executa a busca utilizando os parâmetros especificados e mostra um gráfico com o resultado da métrica para cada
    ponto na grade. Mostra também o resultado do algoritmo de imageamento para uma posição na grade, que o usuário pode
    escolher e mudar.

    Para salvar um gráfico, basta clicar com o botão direito na imagem desejada, e selecionar **Save**.
    """

    def __init__(self, main_window):
        """Construtor da classe.

        Parameters
        ----------
            main_window : :class:`guiqt.gui.MainWindow`
                Janela principal da interface gráfica.
        """
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.main_window = main_window

        # linha para selecionar uma imagem dos valores estimados
        self.infinite_line = pg.InfiniteLine(movable=True)

        # valor e indice da ultima imagem mostrada
        self.ultimo_valor = 0
        self.ultimo_index = 0

        # chaves das imagens geradas
        self.chaves = []

        # dicionario das imagens geradas
        self.images = {}

        # flag para evitar loops
        self.line_changing = False

        self.plot_widget_esq.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plot_widget_esq.customContextMenuRequested.connect(self.show_pop_menu_img_esq)
        self.plot_widget_esq.getPlotItem().setMenuEnabled(False)

        self.plot_widget_dir.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plot_widget_dir.customContextMenuRequested.connect(self.show_pop_menu_img_dir)
        self.plot_widget_dir.getPlotItem().setMenuEnabled(False)

        # cria menu de contexto das imagens
        self.pop_menu_img = QMenu(self)

        # cria opcao para salvar imagem
        menu_save = QAction("Save", self.pop_menu_img)
        self.pop_menu_img.addAction(menu_save)

        # conecta os sinais necessarios
        self.button_action_estimate.clicked.connect(self.estimate_cl)
        self.infinite_line.sigPositionChangeFinished.connect(self.line_changed)
        # self.check_box_envelope.stateChanged.connect(self.envelope_changed)

        # encontra os algoritmos no modulo ``cl_estimators``
        algs = [x for x in dir(cl_estimators) if x.startswith("cl_estimator_")]
        for i in range(len(algs)):
            self.combo_box_estimador.addItem(algs[i][13:])

        # encontra os algoritmos no pacote ``imaging``
        # path = os.path.abspath(imaging.__file__.split('__init__.py')[0])
        # membros = [x[:-3] for x in os.listdir(path) if x[-3:] == '.py' and not x[:2] == '__']
        # for i in range(len(membros)):
            # self.combo_box_alg.addItem(membros[i])
        membros = ['tfm', 'cpwc', 'saft']
        self.combo_box_alg.addItems(membros)

        self.spin_box_primeiro_cl.setRange(1, 99999)
        self.spin_box_ultimo_cl.setRange(0, 99999)
        self.spin_box_passo_cl.setRange(1, 99999)

        self.spin_box_primeiro_cl.setDecimals(5)
        self.spin_box_ultimo_cl.setDecimals(5)
        self.spin_box_passo_cl.setDecimals(0)

        self.spin_box_ultimo_cl.setValue(6500)

        self.spin_box_primeiro_cl.valueChanged.connect(self.spin_box_primeiro_changed)
        self.spin_box_ultimo_cl.valueChanged.connect(self.spin_box_ultimo_changed)

    def file_opened(self):
        """ Função chamada quando um novo aquivo é aberto.
        """
        # abriu outro arquivo
        # faz nada
        pass

    def draw_curve(self, img=None, v_x=None, v_y=None, show_line=False):
        """ Desenha a curva obtida pelo *grid search*.

        Parameters
        ----------
            img : :class:`numpy.ndarray` ou None
                Imagem a ser desenhada, ou None caso deseja desenhar uma curva a partir de pontos.

            v_x : :class:`numpy.ndarray` ou None
                Pontos em x da curva, ou None para desenhar uma imagem ou utilizar a escala padrão.

            v_y : :class:`numpy.ndarray` ou None
                Pontos em y da curva, ou None para dsenhar uma imagem.

            show_line : `bool`
                Flag para desenhar um cursor vertical.
        """
        if img is not None:
            item = pg.ImageView()
            item.setImage(img.T)
            item = item.getImageItem()
        else:
            if v_x is None:
                v_x = range(len(v_y))
            item = pg.PlotDataItem(v_x, v_y)

        self.plot_widget_esq.getPlotItem().clear()
        self.plot_widget_esq.addItem(item)
        try:
            self.plot_widget_esq.getPlotItem().items[0].setLookupTable(self.main_window.lut)
        except AttributeError:
            pass

        if show_line:
            self.line_changing = True
            self.infinite_line.setBounds([v_x[0], v_x[-1]])
            self.plot_widget_esq.addItem(self.infinite_line)
            self.line_changing = False

    def draw_image(self, img=None, v_x=None, v_y=None):
        """ Desenha a imagem selecionada pelo cursor no ``PlotWidget`` com a curva obtida. Também é possível passar uma
        imagem qualquer.

        Parameters
        ----------
            img : :class:`numpy.ndarray` ou None
                Imagem a ser desenhada, ou None caso deseja desenhar uma curva a partir de pontos.

            v_x : :class:`numpy.ndarray` ou None
                Pontos em x da curva, ou None para desenhar uma imagem ou utilizar a escala padrão.

            v_y : :class:`numpy.ndarray` ou None
                Pontos em y da curva, ou None para dsenhar uma imagem.
        """
        if img is not None:
            # if self.check_box_envelope.isChecked():
            #     img = post_proc.envelope(img, -2)
            max = np.max(np.abs(img))
            img = post_proc.normalize(img, image_max=max, image_min=-max)
            img_view = pg.ImageView()
            img_view.setImage(img.T, levels=(0, 1))
            item = img_view.getImageItem()
        else:
            if v_x is None:
                v_x = range(len(v_y))
            item = pg.PlotDataItem(v_x, v_y)

        self.plot_widget_dir.getPlotItem().clear()
        self.plot_widget_dir.addItem(item)
        try:
            self.plot_widget_dir.getPlotItem().items[0].setLookupTable(self.main_window.lut)
        except AttributeError:
            pass

        if img is not None:
            self.plot_widget_dir.getPlotItem().getViewBox().invertY()

    def estimate_cl(self):
        """ Função para estimar a velocidade. Seleciona o algoritmo de imageamento e a métrica, depois realiza
        *grid search* na *thread* da janela principal.
        """
        if self.main_window.img_rect_esq == [0, 0, 0, 0]:
            # nao abriu arquivo
            return

        start = self.spin_box_primeiro_cl.value()
        tol = self.spin_box_passo_cl.value()
        end = self.spin_box_ultimo_cl.value()

        # pega o estimador selecionado
        str_estimador = "cl_estimator_"+self.combo_box_estimador.currentText()
        est_kernel = getattr(cl_estimators, str_estimador)

        # pega o algoritmo de imageamento selecionado
        str_modulo = self.combo_box_alg.currentText()
        modulo = importlib.import_module('.' + str_modulo, 'imaging')
        img_kernel = getattr(modulo, str_modulo + '_kernel')

        d = {'data': self.main_window.dados, 'roi': self.main_window.roi_framework, 'img_func': img_kernel,
             'sel_shot': self.main_window.spin_box_shot.value(),  'a': start, 'b': end, 'tol': tol,
             'metric_func': est_kernel}

        self.images.clear()
        self.chaves.clear()# = np.ndarray([])

        self.main_window.run_in_thread(cl_estimators.gs, d, self.estimate_cl_finished, show_overlay=False)

    def estimate_cl_finished(self, result):
        c_est, est = result

        self.text_browser.setText(f"Estimated speed: {c_est:.2f}m/s")

        dict_sorted = collections.OrderedDict(sorted(est.items()))

        v_x = list(dict_sorted.keys())
        v_y = post_proc.normalize(np.asarray(list(dict_sorted.values())).T[0])
        self.chaves = list(np.asarray(list(dict_sorted.values()), dtype=np.int64).T[1])
        for k in self.chaves:
            self.images[k] = np.abs(self.main_window.dados.imaging_results[k].image)
            self.main_window.dados.imaging_results.pop(k)

        self.draw_curve(v_y=v_y, v_x=v_x, show_line=True)
        self.ultimo_valor = 0
        self.line_changed()

    def line_changed(self):
        """ Chamado quando a posição do cursor é alterada. Desenha a imagem referente à nova velocidade
        selecionada.
        """
        if self.line_changing:
            return

        self.line_changing = True
        # seleciona o valor mais proximo
        v_x = self.plot_widget_esq.getPlotItem().items[0].getData()[0]
        val = min(v_x, key=lambda x: abs(x-self.infinite_line.value()))
        index = np.where(v_x == val)[0][0]

        if val != self.ultimo_valor:
            self.draw_image(self.images[int(self.chaves[index])])

        self.ultimo_valor = val
        self.ultimo_index = index
        self.infinite_line.setValue(val)

        self.line_changing = False

    def change_lut(self, lut):
        """ Troca o mapa de cores da imagem.

        Parameters
        ----------
            lut : `String`
                Nome do mapa de cor escolhido.
        """
        try:
            self.plot_widget_esq.getPlotItem().items[0].setLookupTable(lut)
        except (AttributeError, IndexError):
            pass
        try:
            self.plot_widget_dir.getPlotItem().items[0].setLookupTable(lut)
        except (AttributeError, IndexError):
            pass

    def envelope_changed(self):
        """ Chamado quando o *check box* do envelope é alterado. Redesenha a imagem.
        """
        if len(self.chaves) == 0:
            return
        self.draw_image(self.images[self.chaves[self.ultimo_index]])

    def show_pop_menu_img_esq(self, point):
        """ Mostra um *pop menu*.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.

        """
        action = self.pop_menu_img.exec_(self.plot_widget_esq.mapToGlobal(point))
        self.treat_action(action, 'l')

    def show_pop_menu_img_dir(self, point):
        """ Mostra um *pop menu*.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.

        """
        action = self.pop_menu_img.exec_(self.plot_widget_dir.mapToGlobal(point))
        self.treat_action(action, 'r')

    def treat_action(self, action, pos):
        """ Trata as ações dos menus de contexto das imagens.

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
            if pos == 'l':  # imagem da esquerda
                img = self.plot_widget_esq
            elif pos == 'r':
                img = self.plot_widget_dir
            else:
                return
            self.main_window.save_image(img)

        else:  # muda colormap
            self.main_window.change_lut(action.text())

    def spin_box_primeiro_changed(self, value):
        self.spin_box_ultimo_cl.setMinimum(value)

    def spin_box_ultimo_changed(self, value):
        self.spin_box_primeiro_cl.setMaximum(value)

    def disable(self, b):
        """ Desabilita todos os *widgets* enquanto a *thread* da janela principal está trabalhando.

        Parameters
        ----------
            b : `bool`
                Flag para desativar ou ativar os widgets.
        """
        self.spin_box_ultimo_cl.setDisabled(b)
        self.spin_box_primeiro_cl.setDisabled(b)
        self.spin_box_passo_cl.setDisabled(b)
        self.combo_box_alg.setDisabled(b)
        self.combo_box_estimador.setDisabled(b)
        # self.check_box_envelope.setDisabled(b)
        self.button_action_estimate.setDisabled(b)
