# -*- coding: utf-8 -*-
"""
Módulo ``ShotsSelectionWindow``
===============================

Implementa uma janela para seleção dos disparos a serem carregados.

.. raw:: html

    <hr>

"""

import numpy as np
from guiqt.Utils.ParameterRoot import ParameterRoot
from pyqtgraph import parametertree
from PyQt5 import QtWidgets, QtCore


class ShotsSelectionWindow(QtWidgets.QDialog):
    """ Essa janela permite o usuário selecionar quais disparos de uma inspeção deverão ser carregados.
    Utiliza um ``Utils.ArrayParameter.ArrayParameter`` para permitir ao usuário escolher os disparos.
    """
    def __init__(self, parent=None):
        super(ShotsSelectionWindow, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Shot selection")
        self.setWindowFlags(self.windowFlags() ^ QtCore.Qt.WindowContextHelpButtonHint)
        self.setMinimumSize(300, 95)
        self.setMaximumSize(300, 95)

        self.layout = QtWidgets.QGridLayout()

        self.parameter_tree = parametertree.ParameterTree()
        self.parameter_tree.setHeaderHidden(True)
        self.parameter_root = ParameterRoot()
        self.parameter_sel_shots = self.parameter_root.addChild({'name': 'shots', 'value': np.array([0]),
                                                                 'type': 'ndarray'})
        self.parameter_tree.setParameters(self.parameter_sel_shots)
        self.parameter_tree.setDisabled(True)

        self.check_box_all_shots = QtWidgets.QCheckBox()
        self.check_box_all_shots.setChecked(True)
        self.check_box_all_shots.stateChanged.connect(self.check_box_clicked)
        self.check_box_all_shots.setText("Load all")

        self.button_confirm = QtWidgets.QPushButton()
        self.button_confirm.clicked.connect(self.button_clicked)
        self.button_confirm.setText("Ok")

        self.sel_shot = -1

        self.layout.addWidget(self.parameter_tree, 0, 1, 1, 2)
        self.layout.addWidget(self.check_box_all_shots, 1, 0, 1, 2)
        self.layout.addWidget(self.button_confirm, 2, 0, 1, 2)
        self.setLayout(self.layout)

        self.exec_()

    def check_box_clicked(self):
        if self.check_box_all_shots.isChecked():
            self.parameter_tree.setDisabled(True)
        else:
            self.parameter_tree.setEnabled(True)

    def button_clicked(self):
        if self.check_box_all_shots.isChecked():
            self.sel_shot = None
        else:
            sel_shot = self.parameter_sel_shots.value()
            self.sel_shot = [int(x) for x in sel_shot]

        self.close()
