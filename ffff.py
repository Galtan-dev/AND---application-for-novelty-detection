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
from PyQt5.QtCore import QDate, Qt, pyqtSignal, pyqtSlot
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import numpy as np
import matplotlib.pyplot as plt
import padasip as pa

# hlavní okno
class window_one(QMainWindow):
    signal = pyqtSignal()
    def __init__(self):
        QMainWindow.__init__(self)
        super(window_one, self).__init__()
        self.resize(400, 300)
        self.connectsignal()

    def connectsignal(self):
        sig_1 = 125

        self.window_two = window_two()
        self.signal.connect(self.window_two.signal_check)
        self.signal.emit(sig_1)


# vedlejší okno s nápisem
class window_two(QWidget):
    def __init__(self):
        super(window_two,self).__init__()
        self.resize(400, 300)

    @pyqtSlot()
    def signal_check(self, sig_1):
      print(sig_1)



# zobrazení a exit z aplikace
app = QApplication(sys.argv)
w = window_one()
w.show()
app.exec()