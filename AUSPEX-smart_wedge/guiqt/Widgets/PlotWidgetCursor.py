# -*- coding: utf-8 -*-
"""
Módulo ``PlotWidgetCursor``
===========================

.. raw:: html

    <hr>
    
"""

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore

from framework import post_proc


class PlotWidgetCursor(pg.PlotWidget):
    """ Instancia um ``pyqtgraph.PlotWidget`` contendo duas linhas que formam um cursor.
    """
    def __init__(self, dock_widget, pop_menu=None, nome=None):
        """ Construtor da classe.

        Parameters
        ----------
            dock_widget : :class:`pyqtgraph.dockarea.DockArea` ou *None*.
                *Widget* em que o ``PlotWidgetCursor`` será colocado. Caso não exista, é possível passar *None*.
        """
        super(self.__class__, self).__init__()
        self.nome = nome
        self.pop_menu = pop_menu
        self.img = np.zeros((1, 1))
        self.img_env = np.zeros((1, 1))
        self.min_gain = 0
        self.max_gain = 0
        self.bounding_rect = QtCore.QRectF()
        self.lut = None
        self.v_line = pg.InfiniteLine(angle=90, movable=True)
        self.h_line = pg.InfiniteLine(angle=0, movable=True)
        self.dock_widget = dock_widget
        self.v_line.sigPositionChanged.connect(self.line_changed)
        self.h_line.sigPositionChanged.connect(self.line_changed)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.getPlotItem().setMenuEnabled(False)
        self.envelope = False

    def draw_image(self, img, rect, lut, min=None, max=None):
        """ Desenha uma imagem.

        Parameters
        ----------
            img : :class:`numpy.ndarray`
                Imagem a ser desenhada.
            rect : :class:`pyqtgraph.QtCore.QRectF`
                Limites da imagem.
            lut : :class:`numpy.ndarray`
                *Look up Table* para as cores da imagem.
            min : :class:`float`
                Valor mínimo da imagem a ser desenhada.
            max : :class:`float`
                Valor máximo da imagem a ser desenhada.
        """
        self.img = img
        if rect is None:
            rect = QtCore.QRectF(0, 0, img.shape[1], img.shape[0])
        self.bounding_rect = rect
        self.lut = lut
        img_item = pg.ImageItem()
        img_item.setLookupTable(lut)
        if self.dock_widget.check_box_lock_gain.isChecked():
            min, max = self.dock_widget.get_gain()
        else:
            if max is None:
                max = np.max(np.abs(img))
            if min is None:
                min = -max
        img_item.setImage(post_proc.normalize(img.T, image_max=max, image_min=min), levels=(0, 1))
        img_item.setRect(rect)
        self.min_gain = min
        self.max_gain = max
        self.getPlotItem().clear()
        self.addItem(img_item)
        self.addItem(self.v_line)
        self.v_line.setBounds((rect.left(), rect.right()))
        self.addItem(self.h_line)
        self.h_line.setBounds((rect.top(), rect.bottom()))
        self.getPlotItem().getViewBox().invertY()
        self.envelope = False

    def redraw(self):
        """ Redesenha a imagem atual com os novos parâmetros inseridos neste objeto.
        """
        self.draw_image(self.img, self.bounding_rect, self.lut)

    def line_changed(self):
        """ Chamado quando uma das linhas do cursor é movida.
        """
        z = self.h_line.getYPos()
        x = self.v_line.getXPos()

        x_max = self.bounding_rect.right()
        x_min = self.bounding_rect.left()
        z_max = self.bounding_rect.bottom()
        z_min = self.bounding_rect.top()

        self.dock_widget.change_coords(x, z, (x_min, x_max), (z_min, z_max))
        self.dock_widget.change_amplitude(self.get_amplitude())

        max = np.max(np.abs(self.img))
        min = -max
        self.dock_widget.change_gain(min, max)
        self.dock_widget.focus_changed(self)

    def get_cursor_pos(self):
        """
        Retorna a posição atual do cursor nos limites da imagem.

        Returns
        -------
        :class:`tuple`
            Posições vertical e horizontal do cursor.
        """
        return self.v_line.getXPos(), self.h_line.getYPos()

    def set_cursor_pos(self, px, pz):
        """ Muda as posições do cursor.

        Parameters
        ----------
            px : :class: `float`
                posição horizontal.
            pz : :class: `float`
                posição vertical.
        """
        self.v_line.setPos(px)
        self.h_line.setPos(pz)

    def get_gain(self):
        """
        Retorna o mínimo e máximo da imagem.

        Returns
        -------
        :class:`tuple`
            Menor valor e maior valor da imagem.
        """
        return self.min_gain, self.max_gain

    def get_amplitude(self):
        """
        Retorna o valor do pixel em que o cursor se encontra.

        Returns
        -------
        :class:`float`
            Valor do pixel.
        """
        z = self.h_line.getYPos()
        x = self.v_line.getXPos()

        x_max = self.bounding_rect.right()
        x_min = self.bounding_rect.left()
        i_x = int((x - x_min) / (x_max - x_min) * (self.img.shape[1]))
        i_x = self.img.shape[1] - 1 if i_x >= self.img.shape[1] else i_x
        z_max = self.bounding_rect.bottom()
        z_min = self.bounding_rect.top()
        i_z = int((z - z_min) / (z_max - z_min) * (self.img.shape[0]))
        i_z = self.img.shape[0] - 1 if i_z >= self.img.shape[0] else i_z

        if self.envelope:
            return self.img_env[i_z, i_x]
        return self.img[i_z, i_x]

    def draw_envelope(self):
        """
        Desenha a imagem atual com envelope.
        """
        img = post_proc.envelope(self.img, -2)

        img_item = pg.ImageItem()
        img_item.setLookupTable(self.lut)
        if self.dock_widget.check_box_lock_gain.isChecked():
            min, max = self.dock_widget.get_gain()
        else:
            max = np.max(np.abs(img))
            min = -max
        img_item.setImage(post_proc.normalize(img.T, image_max=max, image_min=min), levels=(0, 1))
        img_item.setRect(self.bounding_rect)
        self.min_gain = min
        self.max_gain = max
        self.getPlotItem().clear()
        self.addItem(img_item)
        self.addItem(self.v_line)
        self.v_line.setBounds((self.bounding_rect.left(), self.bounding_rect.right()))
        self.addItem(self.h_line)
        self.h_line.setBounds((self.bounding_rect.top(), self.bounding_rect.bottom()))
        self.getPlotItem().getViewBox().invertY()
        self.img_env = img
        self.envelope = True

    def show_context_menu(self, point):
        if self.pop_menu is not None:
            action = self.pop_menu.exec_(self.mapToGlobal(point))
            self.dock_widget.treat_action(action, self)
