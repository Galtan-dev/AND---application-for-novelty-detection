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



import numpy as np
import matplotlib.pylab as plt
import padasip as pa

# creation of data
N = 500
x = np.random.normal(0, 1, (N, 4)) # input matrix
v = np.random.normal(0, 0.1, N) # noise
d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target

print(type(d))
print(d.shape)
print(type(x))
print(x.shape)
print(x)

# identification
f = pa.filters.FilterGNGD(n=4, mu=0.1, w="random")
y, e, w = f.run(d, x)

