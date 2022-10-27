import random
import sys
from pathlib import Path
import statistics
import os

from PyQt5.QtWidgets import (QMainWindow,QWidget, QComboBox, QPushButton,\
                             QAction, QLabel, QFileDialog, QTableWidgetItem,\
                             QLineEdit, QTableWidget, QCheckBox, QMessageBox,\
                             QApplication,QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import (QIcon)
from PyQt5.QtCore import QSize, QRect, Qt, pyqtSignal
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import numpy as np
import matplotlib.pyplot as plt
import padasip as pa

# hlavní okno
class window_one(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        super(window_one, self).__init__()
        self.resize(400, 300)
        self.window_two = window_two()
        self.window_two.signal_transmit_data.connect(self.placeholder)

        self.window_two.show()

    def placeholder(self,data: int):
        print(data)

    #     self.connectsignal()
    #
    # def connectsignal(self):
    #     sig_1 = 125
    #
    #     self.window_two = window_two()
    #     #self.signal.connect(self.window_two.signal_check)
    #     self.signal.emit(sig_1)


# vedlejší okno s nápisem
class window_two(QWidget):
    signal_transmit_data = pyqtSignal(int)
    def __init__(self):
        super().__init__()
        self.setMinimumSize(QSize(800, 120))
        self.button = QPushButton(self)
        self.button.setGeometry(QRect(50,50,50,50))
        self.button.clicked.connect(self.push_click)
        self.button.show()
        self.list_window1 = []
    def push_click(self):
        self.signal_transmit_data.emit(100)





# zobrazení a exit z aplikace
app = QApplication(sys.argv)
w = window_one()
w.show()
app.exec()