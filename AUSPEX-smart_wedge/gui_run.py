#!/usr/bin/env python 

"""Script para execução da aplicação GUI."""
import os
import sys
import multiprocessing
from PyQt5 import QtWebEngineWidgets
import guiqt.gui

# Inclusão dos packages do projeto AUSPEX no PYTHONPATH.
sys.path.append(os.path.join(os.path.dirname(__file__), "framework"))
sys.path.append(os.path.join(os.path.dirname(__file__), "guiqt"))
sys.path.append(os.path.join(os.path.dirname(__file__), "imaging"))
sys.path.append(os.path.join(os.path.dirname(__file__), "surface"))
sys.path.append(os.path.join(os.path.dirname(__file__), "parameter_estimation"))

# Importação do módulo da aplicação GUI.


if __name__ == '__main__':
    # Pyinstaller fix
    multiprocessing.freeze_support()
    guiqt.gui.main()
    sys.exit()


