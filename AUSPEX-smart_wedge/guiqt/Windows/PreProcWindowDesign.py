# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PreProcWindowDesign.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_pre_proc_dialog(object):
    def setupUi(self, pre_proc_dialog):
        pre_proc_dialog.setObjectName("pre_proc_dialog")
        pre_proc_dialog.resize(734, 546)
        self.gridLayout = QtWidgets.QGridLayout(pre_proc_dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.combo_box_alg = QtWidgets.QComboBox(pre_proc_dialog)
        self.combo_box_alg.setObjectName("combo_box_alg")
        self.gridLayout.addWidget(self.combo_box_alg, 6, 2, 1, 1)
        self.plot_widget_ascan = PlotWidget(pre_proc_dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plot_widget_ascan.sizePolicy().hasHeightForWidth())
        self.plot_widget_ascan.setSizePolicy(sizePolicy)
        self.plot_widget_ascan.setObjectName("plot_widget_ascan")
        self.gridLayout.addWidget(self.plot_widget_ascan, 1, 0, 1, 3)
        self.parametertree = ParameterTree(pre_proc_dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.parametertree.sizePolicy().hasHeightForWidth())
        self.parametertree.setSizePolicy(sizePolicy)
        self.parametertree.setObjectName("parametertree")
        self.gridLayout.addWidget(self.parametertree, 6, 0, 7, 2)
        self.button_reset = QtWidgets.QPushButton(pre_proc_dialog)
        self.button_reset.setObjectName("button_reset")
        self.gridLayout.addWidget(self.button_reset, 10, 2, 1, 1)
        self.button_apply = QtWidgets.QPushButton(pre_proc_dialog)
        self.button_apply.setObjectName("button_apply")
        self.gridLayout.addWidget(self.button_apply, 9, 2, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.spin_box_sequence = QtWidgets.QSpinBox(pre_proc_dialog)
        self.spin_box_sequence.setObjectName("spin_box_sequence")
        self.gridLayout_2.addWidget(self.spin_box_sequence, 2, 0, 1, 1)
        self.label = QtWidgets.QLabel(pre_proc_dialog)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.spin_box_channel = QtWidgets.QSpinBox(pre_proc_dialog)
        self.spin_box_channel.setObjectName("spin_box_channel")
        self.gridLayout_2.addWidget(self.spin_box_channel, 2, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(pre_proc_dialog)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(pre_proc_dialog)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 2, 1, 1)
        self.spin_box_shot = QtWidgets.QSpinBox(pre_proc_dialog)
        self.spin_box_shot.setObjectName("spin_box_shot")
        self.gridLayout_2.addWidget(self.spin_box_shot, 2, 2, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 2, 0, 1, 3)
        self.button_save = QtWidgets.QPushButton(pre_proc_dialog)
        self.button_save.setObjectName("button_save")
        self.gridLayout.addWidget(self.button_save, 12, 2, 1, 1)
        self.plot_widget_bscan = PlotWidget(pre_proc_dialog)
        self.plot_widget_bscan.setObjectName("plot_widget_bscan")
        self.gridLayout.addWidget(self.plot_widget_bscan, 0, 0, 1, 3)
        self.button_resetall = QtWidgets.QPushButton(pre_proc_dialog)
        self.button_resetall.setObjectName("button_resetall")
        self.gridLayout.addWidget(self.button_resetall, 11, 2, 1, 1)

        self.retranslateUi(pre_proc_dialog)
        QtCore.QMetaObject.connectSlotsByName(pre_proc_dialog)

    def retranslateUi(self, pre_proc_dialog):
        _translate = QtCore.QCoreApplication.translate
        pre_proc_dialog.setWindowTitle(_translate("pre_proc_dialog", "Preprocessing"))
        self.button_reset.setText(_translate("pre_proc_dialog", "Undo last"))
        self.button_apply.setText(_translate("pre_proc_dialog", "Apply"))
        self.label.setText(_translate("pre_proc_dialog", "Sequence"))
        self.label_2.setText(_translate("pre_proc_dialog", "Channel"))
        self.label_3.setText(_translate("pre_proc_dialog", "Shot"))
        self.button_save.setText(_translate("pre_proc_dialog", "Close"))
        self.button_resetall.setText(_translate("pre_proc_dialog", "Reset all"))

from pyqtgraph import PlotWidget
from pyqtgraph.parametertree import ParameterTree
