# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ImagingDesign.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_imaging(object):
    def setupUi(self, imaging):
        imaging.setObjectName("imaging")
        imaging.resize(587, 538)
        self.gridLayout = QtWidgets.QGridLayout(imaging)
        self.gridLayout.setObjectName("gridLayout")
        self.img_tab = QtWidgets.QTabWidget(imaging)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.img_tab.sizePolicy().hasHeightForWidth())
        self.img_tab.setSizePolicy(sizePolicy)
        self.img_tab.setObjectName("img_tab")
        self.gridLayout.addWidget(self.img_tab, 0, 0, 3, 2)

        self.retranslateUi(imaging)
        self.img_tab.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(imaging)

    def retranslateUi(self, imaging):
        _translate = QtCore.QCoreApplication.translate
        imaging.setWindowTitle(_translate("imaging", "Form"))

