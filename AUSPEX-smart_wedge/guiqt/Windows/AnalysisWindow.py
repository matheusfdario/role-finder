# -*- coding: utf-8 -*-
"""
Módulo ``AnalysisWindow``
=========================

Implementa uma janela de análise. Permite visualizar várias imagens, aplicar métricas e alterar os valores máximos e
mínimos das imagens.

.. raw:: html

    <hr>
    
"""

import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets

from framework import post_proc
from guiqt.Windows import AnalysisWindowDesign
from guiqt.Windows.ErrorWindow import ErrorWindow
from guiqt.Widgets.PlotWidgetCursor import PlotWidgetCursor


class AnalysisWindow(AnalysisWindowDesign.Ui_MainWindow):
    """
    Classe da janela de análise. Possui opção para adicionar imagens feitas com o ``ImagingWidget``. Essas imagens
    possuem um cursor que permite ver o valor de um pixel. Também é possível alterar os valores de máximo e mínimo da
    imagem, medir ``API`` e ``CNR``.
    """
    def __init__(self, main_window, tabs):
        """ Construtor da classe

        Parameters
        ----------
            main_window : :class:`guiqt.gui.MainWindow`
                Janela principal da interface.
            tabs : :class:`list`
                Lista com abas do *widget* de imageamento.
        """
        super(self.__class__, self).__init__()
        self.form = QtWidgets.QMainWindow()
        self.setupUi(self.form)
        self.form.show()
        self.form.setWindowTitle('AUSPEX: Analysis')

        self.main_window = main_window

        self.rois = {}

        wavelength = self.main_window.dados.specimen_params.cl/self.main_window.dados.probe_params.central_freq
        self.spin_box_wavelength.setValue(wavelength)

        self.check_box_lock_gain.stateChanged.connect(self.gain_locked)
        self.check_box_envelope.stateChanged.connect(self.envelope_changed)
        self.spin_box_min_gain.editingFinished.connect(self.gain_locked)
        self.spin_box_max_gain.editingFinished.connect(self.gain_locked)
        self.spin_box_x_coord.valueChanged.connect(self.spin_box_coord_changed)
        self.spin_box_z_coord.valueChanged.connect(self.spin_box_coord_changed)
        self.spin_box_wavelength.valueChanged.connect(self.spin_box_wavelength_changed)
        self.combo_box_plot.currentIndexChanged.connect(self.combo_box_changed)
        self.combo_box_cnr_background.currentIndexChanged.connect(self.update_cnr)
        self.combo_box_cnr_foreground.currentIndexChanged.connect(self.update_cnr)
        self.button_add_img.clicked.connect(self.add_tab_from_imaging)
        self.spin_box_min_gain.setDisabled(True)
        self.spin_box_max_gain.setDisabled(True)

        self.pop_menu_img = QtWidgets.QMenu()
        self.pop_menu_img.addAction(QtWidgets.QAction("Save", self.pop_menu_img))
        self.pop_menu_img.addAction(QtWidgets.QAction("Difference", self.pop_menu_img))

        for tab in tabs:
            self.combo_box_add_img.addItem(tab.tab_name)

        self.tabs = tabs

        # fix to prevent crashes when closing all tabs
        self.dock_area.temporary = False

        self.updating = False

    def update_tabs(self, tabs):
        """ Atualiza a lista de abas do *widget* de imageamento.

        Parameters
        ----------
            tabs : :class:`list`
                Lista com abas do *widget* de imageamento.
        """
        self.updating = True
        self.tabs = tabs
        self.combo_box_add_img.clear()
        for tab in tabs:
            if tab.img_proc is not None:
                self.combo_box_add_img.addItem(tab.tab_name)
        self.updating = False

    def add_tab_from_imaging(self):
        """ Chamado quando o usuário seleciona uma aba e clica no botão para adicionar.
        """
        tab_name = self.combo_box_add_img.currentText()
        if tab_name is '':  # no tabs
            return
        tab = None
        for tab in self.tabs:
            if tab_name == tab.tab_name:
                break

        if tab is None:
            return

        if len(tab.img_proc.shape) > 2:
            ErrorWindow("Only 2d images can be added to analysis window")
            return

        rect = QtCore.QRectF(tab.axis_limits[0], tab.axis_limits[1],
                             tab.axis_limits[2] - tab.axis_limits[0],
                             tab.axis_limits[3] - tab.axis_limits[1])

        self.add_image(tab.img_proc, rect, nome=tab.tab_name, roi=tab.roi)

    def add_image(self, img, rect, nome=' ', roi=None):
        """ Cria um ``PlotCursorWidget`` e adiciona na janela.

        Parameters
        ----------
            img : :class:`numpy.ndarray`
                Imagem a ser desenhada.
            rect : :class:`pyqtgraph.QtCore.QRectF`
                Limites da imagem.
            nome : :class:`String`
                Nome do *widget*.
            roi : :class:`framework.data_types.ImagingROI` ou *None*
                ROI da imagem.

        """
        self.updating = True
        pwc = PlotWidgetCursor(self, self.pop_menu_img, nome=nome)
        pwc.draw_image(img, rect, self.main_window.lut)
        if self.check_box_envelope.isChecked():
            pwc.draw_envelope()
        d = pg.dockarea.Dock(nome, closable=True)
        d.addWidget(pwc)
        # removes the detach functionality from the dock
        d.label.mouseDoubleClickEvent = lambda *args: None
        d.sigClosed.connect(self.dock_closed)
        self.dock_area.addDock(d, 'right')
        self.combo_box_plot.addItem(nome, userData=pwc)
        self.combo_box_cnr_background.addItem(nome, userData=pwc)
        self.combo_box_cnr_foreground.addItem(nome, userData=pwc)
        self.rois[pwc] = roi

        pwc.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        pwc.getPlotItem().setMenuEnabled(False)

        self.updating = False

        self.combo_box_changed(self.combo_box_plot.currentIndex())

    def change_coords(self, x, z, range_x=None, range_z=None, force=False):
        """ Muda as posições e limites do cursor na imagem atual.

            Parameters
            ----------
                x : :class:`float`
                    Posição horizontal.
                z : :class:`float`
                    Posição vertical.
                range_x : :class:`tuple` ou *None*
                    Limites no eixo x. Se for *None*, os limites não são alterados.
                range_z : :class:`tuple` ou *None*
                    Limites no eixo z. Se for *None*, os limites não são alterados.
            """
        if self.updating and not force:
            return
        self.updating = True

        if range_x is not None:
            self.spin_box_x_coord.setRange(range_x[0], range_x[1])
        self.spin_box_x_coord.setValue(x)
        if range_z is not None:
            self.spin_box_z_coord.setRange(range_z[0], range_z[1])
        self.spin_box_z_coord.setValue(z)

        self.updating = False

    def change_amplitude(self, value, force=False):
        """ Muda o texto da amplitude.

            Parameters
            ----------
                value : :class:`float`
                    Valor novo da amplitude.
        """
        if self.updating and not force:
            return
        self.updating = True

        #self.line_edit_amplitude.setText(format(format(value, '0.17f'), '.17s'))
        sign = ''
        if value < 0:
            sign = '-'
            value = -value

        self.line_edit_amplitude.setText(sign + ("{0:.{1}e}".format(value, 2)))

        self.updating = False

    def change_gain(self, min, max, force=False):
        """ Muda os limites das imagens.

            Parameters
            ----------
                min : :class:`float`
                    Valor mínimo.
                max : :class:`float`
                    Valor máximo.
        """
        if self.updating and not force:
            return
        self.updating = True

        if not self.check_box_lock_gain.isChecked():
            self.spin_box_min_gain.setValue(min)
            self.spin_box_max_gain.setValue(max)

        self.updating = False

    def get_gain(self):
        """
        Retorna o mínimo e máximo da imagem atual.

        Returns
        -------
        :class:`tuple`
            Menor valor e maior valor da imagem.
        """
        return self.spin_box_min_gain.value(), self.spin_box_max_gain.value()

    def gain_locked(self):
        if self.updating:
            return
        self.updating = True

        if not self.check_box_lock_gain.isChecked():
            self.spin_box_min_gain.setDisabled(True)
            self.spin_box_max_gain.setDisabled(True)
        else:
            self.spin_box_min_gain.setEnabled(True)
            self.spin_box_max_gain.setEnabled(True)

        for i in range(self.combo_box_plot.count()):
            widget = self.combo_box_plot.itemData(i)
            if self.check_box_envelope.isChecked():
                widget.draw_envelope()
            else:
                widget.redraw()

        self.updating = False

        self.combo_box_changed(self.combo_box_plot.currentIndex())

    def focus_changed(self, w):
        if self.updating:
            return
        self.updating = True

        self.combo_box_plot.setCurrentIndex(self.combo_box_plot.findData(w))
        self.update_api(w)

        self.updating = False

    def combo_box_changed(self, index):
        if self.updating:
            return
        self.updating = True

        w = self.combo_box_plot.itemData(index)
        if w is None:  # no widgets in current page
            self.updating = False
            return
        px, pz = w.get_cursor_pos()
        self.change_coords(px, pz, force=True)
        self.change_amplitude(w.get_amplitude(), force=True)
        min, max = w.get_gain()
        self.change_gain(min, max, force=True)

        self.update_api(w)

        self.updating = False

    def spin_box_coord_changed(self):
        if self.updating:
            return
        self.updating = True

        w = self.combo_box_plot.currentData()
        if w is None:  # no widgets in current page
            self.updating = False
            return
        px = self.spin_box_x_coord.value()
        pz = self.spin_box_z_coord.value()

        w.set_cursor_pos(px, pz)
        self.change_amplitude(w.get_amplitude(), force=True)

        self.updating = False

    def spin_box_wavelength_changed(self):
        w = self.combo_box_plot.currentData()
        self.update_api(w)

    def envelope_changed(self):
        for i in range(self.combo_box_plot.count()):
            widget = self.combo_box_plot.itemData(i)
            if self.check_box_envelope.isChecked():
                widget.draw_envelope()
            else:
                widget.redraw()
        self.combo_box_changed(self.combo_box_plot.currentIndex())

    def change_api(self, api):
        """
        Muda o valor mostrado da API.

        Parameters
        ----------
            api : :class:`float`
                Novo valor da API.

        """
        #self.line_edit_api_value.setText(format(format(api, '0.11f'), '.11s'))
        sign = ''
        if api < 0:
            sign = '-'
            api = -api
        self.line_edit_api_value.setText(sign + "{0:.{1}e}".format(api, 2))

    def update_api(self, w):
        """
        Recalcula o valor de API para o *widget* selecionado.

        Parameters
        ----------
            w : :class:`PlotWidgetCursor`
                *Widget* selecionado.
        """
        wavelength = self.spin_box_wavelength.value()
        try:
            roi = self.rois[w]
        except KeyError:  # no widgets in page
            return
        if roi is None:
            self.change_api(0)
            return

        api = post_proc.api(w.img, roi, wavelength)
        self.change_api(api)

    def dock_closed(self, dock):
        """
        Chamado quando um *widget* é fechado.
        """
        self.updating = True
        w = dock.widgets[0]
        self.combo_box_plot.removeItem(self.combo_box_plot.findData(w))
        self.combo_box_cnr_foreground.removeItem(self.combo_box_cnr_foreground.findData(w))
        self.combo_box_cnr_background.removeItem(self.combo_box_cnr_background.findData(w))
        self.rois.pop(w)

        self.updating = False
        if self.combo_box_plot.currentData() is not None:
            self.focus_changed(self.combo_box_plot.currentData())

    def update_cnr(self):
        """
        Atualiza o valor da CNR.
        """
        if self.updating:
            return
        self.updating = True
        foreground = self.combo_box_cnr_foreground.currentData().img
        background = self.combo_box_cnr_background.currentData().img
        if foreground is None or background is None:
            self.updating = False
            return
        cnr = post_proc.cnr(foreground, background)
        #self.line_edit_cnr_value.setText(format(format(cnr, '0.12f'), '.12s'))
        sign = ''
        if cnr < 0:
            sign = '-'
            cnr = -cnr
        self.line_edit_cnr_value.setText(sign + "{0:.{1}e}".format(cnr, 2))

        self.updating = False

    def subtract_images_clicked(self):
        """ Abre janela para escolher duas imagens a serem subtraidas.
        """
        # abre janela para escolher imagens
        SubDialog(self, list(self.rois.keys()))

    def subtract_images(self, img1, img2):
        """ Subtrai duas imagens.
        """
        if img1.shape != img2.shape:
            ErrorWindow("Images with different shapes")
            return
        img = img1 - img2
        self.add_image(img, rect=None, nome='subtraction', roi=None)

    def treat_action(self, action, w=None):
        if action is None:
            return
        elif action.text() == "Save":
            self.main_window.save_image(w)
        elif action.text() == "Difference":
            self.subtract_images_clicked()


class SubDialog(QtWidgets.QDialog):
    """
    Janela de diálogo para selecionar duas imagens a serem subtraídas.
    """
    def __init__(self, analysis_window, tabs):
        super(self.__class__, self).__init__()

        self.setMinimumSize(200, 100)
        self.setMaximumSize(200, 100)

        self.grid_layout = QtWidgets.QGridLayout(self)

        self.analysis_window = analysis_window
        self.combo_box_img1 = QtWidgets.QComboBox(self)
        self.combo_box_img2 = QtWidgets.QComboBox(self)
        for tab in tabs:
            self.combo_box_img1.addItem(tab.nome, tab)
            self.combo_box_img2.addItem(tab.nome, tab)

        self.button = QtWidgets.QPushButton("Ok", self)
        self.button.clicked.connect(self.button_clicked)

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setText('-')

        self.label.setMaximumWidth(10)

        self.grid_layout.addWidget(self.combo_box_img1, 0, 0, 1, 1)
        self.grid_layout.addWidget(self.label, 0, 1, 1, 1)
        self.grid_layout.addWidget(self.combo_box_img2, 0, 2, 1, 1)
        self.grid_layout.addWidget(self.button, 1, 0, 1, 3)

        # remove botao '?'
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.exec()

    def button_clicked(self):
        img1 = self.combo_box_img1.currentData().img
        img2 = self.combo_box_img2.currentData().img

        self.analysis_window.subtract_images(img1, img2)
        self.close()
