# -*- coding: utf-8 -*-
"""
Módulo ``ImagingWidget``
========================

.. raw:: html

    <hr>

"""

import importlib
import os

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMenu, QAction

import imaging
from framework.data_types import DataInsp, ImagingROI
from guiqt.Widgets import ImagingWidgetDesign
from guiqt.Widgets.ImagingTabWidget import ImagingTabWidget


class ImagingWidget(QtWidgets.QWidget, ImagingWidgetDesign.Ui_imaging):
    """
    Instancia um *widget* criado através do ``Designer`` que contem a aba com as funcionalidades de imageamento do
    *framework*.

    Permite executar os algoritmos de imageamento presentes no *framework*. Outros algoritmos podem ser adicionados,
    desde que sigam o padrão estabelecido.

    Identifica as métricas presentes em ``parameter_estimation/cl_estimators.py``, desde que seja seguido o padrão
    estabelecido.

    A ROI e o disparo são definidos na janela principal. Os parâmetros podem ser editados em uma lista.
    Os resultados são apresentos em abas.

    Para salvar um gráfico, basta clicar com o botão direito na imagem desejada, e selecionar **Save**.

    O botão de envelope realiza uma operação de pós-processamento, e pode ser selecionado a qualquer momento da execução
    da interface.
    """

    def __init__(self, main_window):
        """ Construtor da classe.
        Inicializa as funções de imageamento na pasta ``imaging`` que possuem a flag de inicialização `initialize`
        com valor ``True``.

        Parameters
        ----------
            main_window : :class:`guiqt.gui.MainWindow`
                Janela principal da interface gráfica.
        """
        super(self.__class__, self).__init__()
        self.setupUi(self)

        self.main_window = main_window

        self.algoritmos = []

        # cria menu de contexto para fechar as abas das imagens da direita
        self.pop_menu_img_tab = QMenu(self)
        self.pop_menu_img_tab.addAction(QAction('close', self))

        # cria corner widget para abrir novas abas
        self.corner_widget = QtWidgets.QToolButton()
        self.corner_widget.setText("+")
        self.corner_widget.clicked.connect(self.new_tab)

        self.img_tab.setCornerWidget(self.corner_widget)

        # cria menu de contexto das imagens
        self.pop_menu_img = QMenu(self)

        # cria opcao para salvar imagem
        menu_save = QAction("Save", self.pop_menu_img)
        self.pop_menu_img.addAction(menu_save)

        # cria menu da aba
        self.img_tab.tabBar().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.img_tab.tabBar().customContextMenuRequested.connect(self.show_pop_menu_img_tab)

        self.tabs = []

        # encontra os algoritmos e inicializa os que usam numba
        self.main_window.run_in_thread(self.pre_load_algs, {}, self.pre_load_algs_finished)

    def pre_load_algs(self):
        """ Inicializa os algoritmos de imageamento. Cria ROI e ``DataInsp`` pequenos para executar os algoritmos. Útil
        para algoritmos que utilizam ``Numba``, que irá realizar toda a compilação durante a abertura da interface.
        """
        self.corner_widget.setCheckable(False)
        self.update_algs()
        data_insp_dummy = DataInsp()
        data_insp_dummy.inspection_params.type_insp = 'contact'
        data_insp_dummy.inspection_params.water_path = 10
        data_insp_dummy.ascan_data[1500, 1:-1, 1:-1, :] = 1
        data_insp_dummy.ascan_data[1510, 0, 0, :] = 1
        data_insp_dummy.ascan_data[1510, -1, -1, :] = 1
        roi_dummy = ImagingROI(h_len=2, w_len=2)

        for alg in self.algoritmos:
            modulo = importlib.import_module('.' + alg, 'imaging')
            try:
                if modulo.initialize is True:
                    alg_start = getattr(modulo, alg + '_kernel')
                    alg_start(data_insp_dummy, roi=roi_dummy)
            except AttributeError:
                pass

        return True

    def pre_load_algs_finished(self, ok):
        if ok:
            self.corner_widget.setCheckable(True)
            self.new_tab()

    def file_opened(self, pickle=None):
        """ Função chamada pela janela principal quando um novo arquivo é carregado ou um resultado salvo pelo
        ``pickle`` é aberto.

        Parameters
        ----------
            pickle : `dict` ou None
                Dicionário composto por pares chave:`framework.data_types.ImagingResult`.
        """
        # # remove as imagens anteriores
        # self.img_tab.clear()
        # self.tabs.clear()
        # self.new_tab()
        # self.update_algs()

        for index in range(self.img_tab.count()):
            try:
                self.tabs.pop()
            except IndexError:
                pass
            self.img_tab.removeTab(0)

        if len(list(self.main_window.dados.imaging_results.keys())) == 0:
            self.new_tab()

        for k in list(self.main_window.dados.imaging_results.keys()):
            self.new_tab()
            self.tabs[self.img_tab.currentIndex()].action_alg_finished(k)

        for index in range(self.img_tab.count()):
            self.tabs[index].disable(False)

        if pickle is None:
            return

        for result in pickle:
            roi = pickle[result].roi
            self.new_tab()
            self.tabs[self.img_tab.currentIndex()].img_proc = pickle[result].image
            rect = QtCore.QRectF(roi.coord_ref[0, 0], roi.coord_ref[0, 2], roi.width, roi.height)
            self.tabs[self.img_tab.currentIndex()].draw(pickle[result].image, rect=rect)
            self.tabs[self.img_tab.currentIndex()].envelope_changed()
            self.tabs[self.img_tab.currentIndex()].key = result
            self.img_tab.setTabText(self.img_tab.currentIndex(), pickle[result].description)

        for index in range(self.img_tab.count()):
            self.tabs[index].disable(False)

    def show_pop_menu_img_tab(self, point):
        """ Mostra o `pop menu` para fechar uma aba e implementa o fechamento da aba.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.
        """
        action = self.pop_menu_img_tab.exec_(self.img_tab.tabBar().mapToGlobal(point))
        if action is not None and action.text() == 'close':
            index = self.img_tab.tabBar().tabAt(point)
            try:
                self.main_window.dados.imaging_results.pop(self.tabs[index].key)
            except KeyError:
                pass
            self.tabs.pop(index)
            self.img_tab.removeTab(index)

            if self.img_tab.count() == 0:
                self.new_tab()

    def show_pop_menu_img_dir(self, point):
        """ Mostra o `pop menu` da imagem direita.

        Parameters
        ----------
            point : :class:`PyQt5.QtCore.QPoint`
                Posição em que o menu irá abrir.
        """
        action = self.pop_menu_img.exec_(self.img_tab.mapToGlobal(point))
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
            if pos == 'r':  # imagem atual da direita
                img = self.img_tab.widget(self.img_tab.currentIndex())
            else:
                return
            self.main_window.save_image(img)

        else:  # muda colormap
            self.main_window.change_lut(action.text())

    def change_lut(self, lut):
        """ Muda o mapa de cor das imagens. Chamado pela janela principal, e não pelo ``treat_action()``.

        Parameters
        ----------
            lut : `list`
                `Look up table` escolhida.
        """
        for tab in self.tabs:
            tab.change_lut(lut)

    def new_tab(self):
        """ Cria uma nova aba do tipo ``TabWidget``
        """
        index = self.img_tab.count()
        widget = QtWidgets.QWidget()
        # cria uma aba com um widget padrao e depois coloca o widget da aba de imageamento
        self.img_tab.insertTab(index, widget, "new tab")
        tab_widget = ImagingTabWidget(widget, self.algoritmos, self.main_window, self, self.img_tab, index)
        self.tabs.insert(index, tab_widget)
        if self.tabs[self.img_tab.currentIndex()].check_box_envelope.isChecked():
            tab_widget.check_box_envelope.click()

        alg_index = self.tabs[self.img_tab.currentIndex()].box_alg.currentIndex()
        tab_widget.box_alg.setCurrentIndex(alg_index)

        self.corner_widget.setChecked(False)

        self.img_tab.setCurrentIndex(index)

    def disable(self, b=True):
        """ Desativa ou ativa novamente todas as abas e trava o botão para criar novas abas.

        Parameters
        ----------
            b : `bool`
                Flag para desativar ou ativar os widgets.
        """
        self.corner_widget.setDisabled(b)
        for tab in self.tabs:
            tab.disable(b)


    def disable_surf(self, b=True):
        """ Desativa ou ativa novamente todas as opções de surface.

        Parameters
        ----------
            b : `bool`
                Flag para desativar ou ativar os widgets.
        """
        for tab in self.tabs:
            tab.disable_surf(b)

    def update_algs(self):
        """ Encontra os algoritmos no pacote ``imaging``.
        """
        path = os.path.abspath(imaging.__file__.split('__init__.py')[0])
        self.algoritmos = [x[:-3] for x in os.listdir(path) if x[-3:] == '.py' and not x[:2] == '__']
        self.algoritmos.sort()
