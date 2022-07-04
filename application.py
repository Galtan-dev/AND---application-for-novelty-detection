"""
GUI for adaptive filters testing.
"""
import sys
from pathlib import Path
import statistics

from PyQt5.QtWidgets import (QMainWindow, QComboBox, QPushButton,\
                             QAction, QLabel, QFileDialog, QTableWidgetItem,\
                             QLineEdit, QTableWidget, QMessageBox, QApplication)
from PyQt5.QtGui import (QIcon)
from PyQt5 import QtCore
from PyQt5 import QtWidgets

import numpy as np
import matplotlib.pyplot as plt
import padasip as pa


# pylint: disable=too-many-instance-attributes
# pylint: disable=c-extension-no-member

class Detekce(QMainWindow):
    """
    Popis
    """
    def __init__(self):
        QMainWindow.__init__(self)

        self.matrix_shape = None
        self.transposed_matrix = None
        self.filter_output = None
        self.filter_parametrs = None
        self.filter_error = None
        self.input_data = None
        self.input_desired_data = None
        self.output_detection_tool = None
        self.learning_rate = None
        self.filter_name = None
        self.detection_name = None
        self.mean = None
        self.standart_deviation = None
        self.variance = None

        self.init_ui()

    def init_ui(self):
        """
        Second main object. There is set geometry of main window
        and it refers to main grafical elements.

        :return: None
        """
        self.combo_1 = QComboBox(self)
        self.combo_2 = QComboBox(self)
        self.label_1 = QtWidgets.QLabel(self)
        self.label =QtWidgets.QLabel(self)
        self.label_2 = QtWidgets.QLabel(self)
        self.setGeometry(100, 100, 900, 580)  # 700
        self.setWindowTitle("AND - Application for novelty detection")
        self.label_I()
        self.tabulka()
        self.menugraf()
        self.popisky()
        self.combobox()
        self.lspeed()
        self.show()

    def combobox(self):
        """
        This object add items into combobox of filters
        so we can choose filter by another object
        :return: None
        """
        self.combo_1.addItem("Choose filter")
        self.combo_1.addItem("SSLMS")
        self.combo_1.addItem("RLS")
        self.combo_1.addItem("NSSLMS")
        self.combo_1.addItem("AP")
        self.combo_1.addItem("GNGD")
        self.combo_1.addItem("NLMF")
        self.combo_1.addItem("NLMS")
        self.combo_1.addItem("LMS")
        self.combo_1.addItem("LMF")
        self.combo_1.setGeometry(550, 80, 130, 25)

        self.combo_2.addItem("Choose detection")
        self.combo_2.addItem("LE")
        self.combo_2.addItem("ELBND")
        self.combo_2.setGeometry(550, 110, 130, 25)

        self.spoust = QPushButton("Intr.speed detection", self)
        self.spoust.setGeometry(550, 495, 130, 25)
        self.spoust.clicked.connect(self.node_2)

        self.spoust = QPushButton("One speed detection", self)
        self.spoust.setGeometry(550, 525, 130, 25)
        self.spoust.clicked.connect(self.node_1)

    def node_2(self):
        """
        Node for button interval speed detection, when you click
        on that buton you start this node which refers
        to individual objects
        :return: None
        """
        bottom_interval = float(self.textbox1.text())
        top_interval = float(self.textbox1s.text())
        step = float(self.textbox1n.text())
        for self.learning_rate in np.arange(bottom_interval,\
                                            top_interval, step):
            self.vyber()
            self.vyberfiltr()
            self.vyberdet()
            self.grafy()

    def node_1(self):
        """
        Node for button one speed detection, when you click
        on that buton you start this node which refers
        to individual objects
        :return: None
        """
        self.lspeedchoose()
        self.vyber()
        self.vyberfiltr()
        self.vyberdet()
        self.grafy()

    def vyberfiltr(self):
        """
        This object choose which adaptive filter we use.
        :return: None
        """
        filt = self.combo_1.currentText()
        if filt == "SSLMS":
            self.filterSSLMS()
        if filt == "RLS":
            self.filterRLS()
        elif filt == "NSSLMS":
            self.filterNSSLMS()
        elif filt == "AP":
            self.filterAP()
        elif filt == "GNGD":
            self.filterGNGD()
        elif filt == "NLMF":
            self.filterNLMF()
        elif filt == "NLMS":
            self.filterNLMS()
        elif filt == "LMS":
            self.filterLMS()
        elif filt == "LMF":
            self.filterLMF()

    def vyberdet(self):
        """
        This object choose which detection tool we use.
        :return: None
        """
        det = self.combo_2.currentText()
        if det == "LE":
            self.detLE()
        elif det == "ELBND":
            self.detELBND()

    def menufil(self):
        """
        Object which define options of filter and add them into combobox.
        :return: None
        """
        self.statusBar()
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Filters")

        sslms = QAction(QIcon("WWW.jpg"), "SSLMS", self)
        sslms.setShortcut("Ctrl+F")
        sslms.setStatusTip("Sign-sign Least-mean-squares (SSLMS)")
        sslms.triggered.connect(self.filterSSLMS)

        rls = QAction(QIcon("WWW.jpg"), "RLS", self)
        rls.setShortcut("Ctrl+R")
        rls.setStatusTip("Recursive Least Squares (RLS)")
        rls.triggered.connect(self.filterRLS)

        nsslms = QAction(QIcon("WWW.jpg"), "NSSLMS", self)
        nsslms.setShortcut("Ctrl+Q")
        nsslms.setStatusTip("Normalized Sign-sign Least-mean-squares (NSSLMS)")
        nsslms.triggered.connect(self.filterNSSLMS)

        ap_filter = QAction(QIcon("WWW.jpg"), "AP", self)
        ap_filter.setShortcut("Ctrl+A")
        ap_filter.setStatusTip("Affine Projection (AP)")
        ap_filter.triggered.connect(self.filterAP)

        gngd = QAction(QIcon("WWW.jpg"), "GNGD", self)
        gngd.setShortcut("Ctrl+G")
        gngd.setStatusTip("Generalized Normalized Gradient Descent (GNGD)")
        gngd.triggered.connect(self.filterGNGD)

        nlms = QAction(QIcon("WWW.jpg"), "NLMS", self)
        nlms.setShortcut("Ctrl+N")
        nlms.setStatusTip("Normalized Least-mean-squares (NLMS)")
        nlms.triggered.connect(self.filterNLMS)

        nlmf = QAction(QIcon("WWW.jpg"), "NLMF", self)
        nlmf.setShortcut("Ctrl+F")
        nlmf.setStatusTip("Normalized Least-mean-fourth (NLMF)")
        nlmf.triggered.connect(self.filterNLMF)

        lms = QAction(QIcon("WWW.jpg"), "LMS", self)
        lms.setShortcut("Ctrl+S")
        lms.setStatusTip("Least-mean-fourth (LMS)")
        lms.triggered.connect(self.filterLMS)

        lmf = QAction(QIcon("WWW.jpg"), "LMF", self)
        lmf.setShortcut("Ctrl+M")
        lmf.setStatusTip("Least-mean-fourth (LMF)")
        lmf.triggered.connect(self.filterLMF)

        file_menu.addAction(sslms)
        file_menu.addAction(rls)
        file_menu.addAction(ap_filter)
        file_menu.addAction(nsslms)
        file_menu.addAction(gngd)
        file_menu.addAction(nlms)
        file_menu.addAction(nlmf)
        file_menu.addAction(lms)
        file_menu.addAction(lmf)

    def menu_detection(self):
        """
        Object which define options of detection tool and add them into combobox.
        :return: None
        """
        self.statusBar()
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Detection")

        elbnd_detection = QAction(QIcon("WWW.jpg"), "ELBND", self)
        elbnd_detection.setShortcut("Ctrl+1")
        elbnd_detection.setStatusTip("Error and Learning Based Novelty Detection (ELBND)")
        elbnd_detection.triggered.connect(self.detELBND)

        le_detection = QAction(QIcon("WWW.jpg"), "LE", self)
        le_detection.setShortcut("Ctrl+2")
        le_detection.setStatusTip("Learning Entropy (LE)")
        le_detection.triggered.connect(self.detLE)

        file_menu.addAction(elbnd_detection)
        file_menu.addAction(le_detection)

    def popisky(self):
        """
        Object which define labels.
        :return: None
        """
        labelfilter = QLabel(self)
        labelfilter.setText("Learning speed")
        labelfilter.setGeometry(550, 225, 130, 25)

        labellenghtfil = QLabel(self)
        labellenghtfil.setText("Filter lenght")
        labellenghtfil.setGeometry(550, 175, 130, 25)

        labelfilename = QLabel(self)
        labelfilename.setText("Filename")
        labelfilename.setGeometry(550, 275, 130, 25)

    def statupgr(self):
        """
        Object which update statistic in label as text and again define label.
        :return: None
        """
        mean_multiplied = 1000 * self.mean
        mean_back = (round(mean_multiplied)) / 1000
        mean_round = str(mean_back)
        self.label.setText("Mean:" + mean_round)
        self.label.setStyleSheet("font-weight:bold")
        self.label.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)

        variance_multiplied = 1000 * self.variance
        variance_back = (round(variance_multiplied)) / 1000
        variance_round = str(variance_back)
        self.label_1.setText("Variance:" + variance_round)
        self.label_1.setStyleSheet("font-weight:bold")
        self.label_1.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)

        deviation_multiplied = 1000 * self.standart_deviation
        deviation_back = (round(deviation_multiplied)) / 1000
        deviation_round = str(deviation_back)
        self.label_2.setText("Standart deviation:" + deviation_round)
        self.label_2.setStyleSheet("font-weight:bold")
        self.label_2.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)

    def statokno(self):
        """
        Object which calculate statistic.
        :return: None
        """
        stlist = np.ndarray.tolist(self.output_detection_tool)
        self.mean = statistics.mean(stlist)
        self.standart_deviation = statistics.stdev(stlist)
        self.variance = statistics.variance(stlist)
        print(self.mean)
        print(self.standart_deviation)
        print(self.variance)


    def label_I(self):
        """
        Object which define statistic labels for the first time.
        :return: None
        """
        self.label.setText("Mean: None")
        self.label.setStyleSheet("font-weight:bold")
        self.label.setStyleSheet("font-size: 10pt")
        self.label.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)
        self.label.update()

        self.label_1.setText("Variance: None")
        self.label_1.setStyleSheet("font-weight:bold")
        self.label_1.setStyleSheet("font-size: 10pt")
        self.label_1.setGeometry(QtCore.QRect(700, 60, 250, 50))
        self.setGeometry(100, 100, 900, 580)
        self.label_1.update()

        self.label_2.setText("Standart deviation: None")
        self.label_2.setStyleSheet("font-weight:bold")
        self.label_2.setStyleSheet("font-size: 10pt")
        self.label_2.setGeometry(QtCore.QRect(700, 80, 250, 50))
        self.setGeometry(100, 100, 900, 580)
        self.label_2.update()

    def menugraf(self):
        """
        Object which define buttons open a file, open and
        save parametrs and define toolbar named tools.
        :return: None
        """
        opfil = QAction(QIcon("WWW.jpg"), "open", self)
        opfil.setShortcut("Ctrl+9")
        opfil.setStatusTip("Open a file")
        opfil.triggered.connect(self.loading)

        ofiltl = QPushButton("Open a file", self)
        ofiltl.setGeometry(550, 50, 130, 25)
        ofiltl.clicked.connect(self.loading)

        namfil = QPushButton("Save parametrs", self)
        namfil.setGeometry(550, 330, 130, 25)
        namfil.clicked.connect(self.saveparametrs)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Tools")
        file_menu.addAction(opfil)

    def loading(self):
        basefile = str(Path.home())
        jmeno = QFileDialog.getOpenFileName(self, "open file", basefile)
        # hodnoty = open(jmeno[0], "r")
        with open(jmeno[0], "r", encoding="utf-8") as hodnoty:
            mat1 =(np.genfromtxt(hodnoty, delimiter=",", skip_header=0))
        loading_matrix_shape = mat1.shape
        if len(loading_matrix_shape) == 1:
            loading_matrix_length = (loading_matrix_shape[0], 1)
        else:
            loading_matrix_length = loading_matrix_shape
        self.table_widget.setRowCount(loading_matrix_length[0])
        self.table_widget.setColumnCount(loading_matrix_length[1])
        for i in range(0, loading_matrix_length[0]):
            for j in range(0, loading_matrix_length[1]):
                try:
                    if len(loading_matrix_shape) == 1:
                        self.table_widget.setItem(i, j, QTableWidgetItem(str(mat1[i])))
                    else:
                        self.table_widget.setItem(i, j, QTableWidgetItem(str(mat1[i, j])))
                except Exception as ex:
                    print(ex)
        self.matrix_shape = loading_matrix_length
        self.statusBar().showMessage("Files are loaded")

    def vyber(self):
        alt = self.table_widget.selectedItems()
        selection_list = []
        selection_list_shape = self.matrix_shape[0]
        print(selection_list_shape)

        for item in alt:
            selection_list.append(float(item.text()))
        selection_matrix = np.asarray(selection_list)
        selection_matrix_shape = selection_matrix.shape[0]
        reshape_parametr = selection_matrix_shape / selection_list_shape
        try:
            reshaped_selection_matrix = np.reshape(selection_matrix, \
                                                   (int(reshape_parametr), selection_list_shape))
            transposed_selection_matrix = reshaped_selection_matrix.T
            self.transposed_matrix = transposed_selection_matrix
        except Exception as ex:
            print(ex)
        self.statusBar().showMessage("Input column are selected")

    def uprava(self):
        try:
            trans_matrix = self.transposed_matrix
            shape = self.matrix_shape[0]

            filter_length = int(self.textbox3.text())

            desired_data_beginning = []
            input_data_begining = []

            for i in range(filter_length, shape, 1):
                input_data_begining = np.append(input_data_begining, trans_matrix[i])
            for i in range(0, shape - filter_length, 1):
                desired_data_beginning = np.append(desired_data_beginning, trans_matrix[i])
            desired_data_in_progres = np.asarray([desired_data_beginning])
            input_data_in_progres = np.asarray([input_data_begining])
            self.input_desired_data = desired_data_in_progres.T
            self.input_data = input_data_in_progres.T

        except:
            trans_matrix = self.transposed_matrix
            print(trans_matrix)
            shape = self.matrix_shape[0]
            desired_data_beginning = []
            input_data_begining = []

            for i in range(5, shape, 1):
                input_data_begining = np.append(input_data_begining, trans_matrix[i])
            for i in range(0, shape - 5, 1):
                desired_data_beginning = np.append(desired_data_beginning, trans_matrix[i])
            desired_data_in_progres = np.asarray([desired_data_beginning])
            input_data_in_progres = np.asarray([input_data_begining])
            self.input_desired_data = desired_data_in_progres.T
            self.input_data = input_data_in_progres.T

    def lspeed(self):
        self.textbox1 = QLineEdit(self)
        self.textbox1.setGeometry(550, 250, 30, 25)
        self.textbox1s = QLineEdit(self)
        self.textbox1s.setGeometry(600, 250, 30, 25)
        self.textbox1n = QLineEdit(self)
        self.textbox1n.setGeometry(650, 250, 30, 25)


        self.textbox2 = QLineEdit(self)
        self.textbox2.setGeometry(550, 300, 130, 25)
        self.textbox3 = QLineEdit(self)
        self.textbox3.setGeometry(550, 200, 130, 25)
        self.show()

    def saveparametrs(self):
        try:
            spn = str(self.textbox1.text())
            name = str(self.textbox2.text())
            stdev = str(self.standart_deviation)
            variance = str(self.variance)
            mean = str(self.mean)
            sr = str(self.textbox3.text())
            art = {"Learning rate": spn, "Filter length": sr, "Filter": self.filter_name,\
                   "Detection tool": self.detection_name, "Standart deviation": stdev,\
                   "Variance": variance, "Mean": mean}
            tra=repr(art)
            f = open(f"{name}.txt", "w+")
            f.write(tra)
            self.statusBar().showMessage("Parametrs save")
            f.close()
        except Exception as ex:
            print(ex)
            art = {"Learning rate": self.learning_rate, "Filter length": sr,\
                   "Filter": self.filter_name,"Detection tool": self.detection_name,\
                   "Standart deviation": stdev,\
                   "Variance": variance, "Mean": mean}
            tra = repr(art)
            f = open("Detection parametrs","w+")
            f.write(tra)
            f.close()

    def lspeedchoose(self):
        try:
            speedvalue = float(self.textbox1.text())
            self.learning_rate = speedvalue
        except Exception as ex:
            print(ex)
            self.learning_rate = 1

    def filterSSLMS(self):
        self.uprava()
        self.filter_name = "SSLMS"
        try:
            f = pa.filters.FilterSSLMS(n=1, mu=self.learning_rate, w="zeros")
            output, error, weights = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("SSLMS filter aplicated")

    def filterRLS(self):
        self.uprava()
        self.filter_name = "RLS"
        try:
            f = pa.filters.FilterRLS(n=1, mu=self.learning_rate, w="zeros")
            output, error, weights = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("RLS filter aplicated")

    def filterNSSLMS(self):
        self.uprava()
        self.filter_name = "NSSLMS"
        try:
            f = pa.filters.FilterNSSLMS(n=1, mu=self.learning_rate, w="zeros")
            output, error, weights = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("NSSLMS filter aplicated")

    def filterAP(self):
        self.uprava()
        self.filter_name = "AP"
        try:
            f = pa.filters.FilterAP(n=1, mu=self.learning_rate, w="zeros")
            output, error, weights = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("AP filter aplicated")

    def filterNLMS(self):
        self.uprava()
        self.filter_name = "NLMS"
        try:
            f = pa.filters.FilterNLMS(n=1, mu=self.learning_rate, w="zeros")
            y, e, w = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = y
        self.filter_parametrs = w
        self.filter_error = e
        self.statusBar().showMessage("NLMS filter aplicated")

    def filterGNGD(self):
        self.uprava()
        self.filter_name = "GNGD"
        try:
            f = pa.filters.FilterGNGD(n=1, mu=self.learning_rate, w="zeros")
            y, e, w = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = y
        self.filter_parametrs = w
        self.filter_error = e
        self.statusBar().showMessage("GNGD filter aplicated")

    def filterNLMF(self):
        self.uprava()
        self.filter_name = "NLMF"
        try:
            f = pa.filters.FilterNLMF(n=1, mu=self.learning_rate, w="zeros")
            y, e, w = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = y
        self.filter_parametrs = w
        self.filter_error = e
        self.statusBar().showMessage("NLMF filter aplicated")

    def filterLMS(self):
        self.uprava()
        self.filter_name = "LMS"
        try:
            f = pa.filters.FilterLMS(n=1, mu=self.learning_rate, w="zeros")
            y, e, w = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = y
        self.filter_parametrs = w
        self.filter_error = e
        self.statusBar().showMessage("LMS filter aplicated")

    def filterLMF(self):
        self.uprava()
        self.filter_name = "LMF"
        try:
            f = pa.filters.FilterLMF(n=1, mu=self.learning_rate, w="zeros")
            y, e, w = f.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = y
        self.filter_parametrs = w
        self.filter_error = e
        self.statusBar().showMessage("LMF filter aplicated")

    def detELBND(self):
        self.detection_name = "ELBND"
        elbnd = pa.detection.ELBND(self.filter_parametrs, self.filter_error, function="max")
        self.output_detection_tool = elbnd
        print(elbnd)
        self.statusBar().showMessage("ELBND detection aplicated")
        self.statokno()
        self.statupgr()

    def detLE(self):
        self.detection_name = "LE"
        le = pa.detection.learning_entropy(self.filter_parametrs, m=30, order=1)
        n = le.shape
        LE = np.reshape(le, (n[0],))
        print(le)
        print(LE)
        self.output_detection_tool = LE
        self.statusBar().showMessage("LE detection aplicated")
        self.statokno()
        self.statupgr()

    def tabulka(self):
        self.table_widget = QTableWidget(self)
        self.table_widget.setRowCount(20)
        self.table_widget.setColumnCount(5)
        self.table_widget.setGeometry(25, 50, 500, 500)

    def grafy(self):

        fig, axs = plt.subplots(4, 1)
        axs[0].plot(self.filter_error)
        axs[1].plot(self.filter_output)
        axs[2].plot(self.filter_parametrs)
        axs[3].plot(self.output_detection_tool)

        # axs[0].subtitle("Filter error for every sample")
        # axs[1].subtitle("Output value")
        # axs[2].subtitle("History of all weights")
        # axs[3].subtitle("Detection values")

        axs[0].set_title("Filter error for every sample")
        axs[1].set_title("Output value")
        axs[2].set_title("History of all weights")
        axs[3].set_title("Detection values")

        axs[0].set_xlabel("$k [-]$")
        axs[1].set_xlabel("$k [-]$")
        axs[2].set_xlabel("$k [-]$")
        axs[3].set_xlabel("$k [-]$")

        axs[0].set_ylabel("$e [-]$")
        axs[1].set_ylabel("$y [-]$")
        axs[2].set_ylabel("$w [-]$")
        axs[3].set_ylabel("$det [-]$")

        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        axs[3].grid(True)

        fig.tight_layout()
        plt.show()
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Window Close',\
                                     'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


def main():
    app = QApplication(sys.argv)
    ex = Detekce()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
