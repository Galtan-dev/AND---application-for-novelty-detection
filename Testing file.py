import sys
from pathlib import Path
import statistics
import os

from PyQt5.QtWidgets import (QMainWindow, QComboBox, QPushButton,\
                             QAction, QLabel, QFileDialog, QTableWidgetItem,\
                             QLineEdit, QTableWidget, QCheckBox, QMessageBox,\
                             QApplication)
from PyQt5.QtGui import (QIcon)
from PyQt5.QtCore import QDate, QTime, QDateTime, Qt
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import numpy as np
import matplotlib.pyplot as plt
import padasip as pa
# Import the os module



x = float(0.36514)

d = str(x)
print(d)
