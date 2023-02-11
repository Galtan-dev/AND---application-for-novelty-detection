"""
GUI for adaptive filters testing.
"""

import sys
from pathlib import Path
import statistics
import os

from PyQt5.QtWidgets import (QMainWindow,QWidget, QComboBox, QPushButton,\
                             QAction, QLabel, QFileDialog, QTableWidgetItem,\
                             QLineEdit, QTableWidget, QCheckBox, QMessageBox,\
                             QApplication, QVBoxLayout, QScrollArea)
from PyQt5.QtGui import (QIcon)
from PyQt5.QtCore import Qt, pyqtSignal
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
        super(Detekce, self).__init__()
        self.selected_columns = None

        self.selection_window = SubWindow()
        self.selection_window.signal_proces.connect(self.column_selection_holder_one)

        self.num_of_parametrs = None
        self.input_columns_filter = []
        self.desired_columns_filter = []
        self.loading_matrix_shape =None
        self.loading_matrix = None
        self.column_count = None
        self.new_input_arary_reshaped = None
        self.testlist = None
        self.directory_2 = None
        self.path_3 = None
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
        self.directory_3 = None
        self.parent_dir_3 = None
        self.label_savefig = None
        self.mean_absolute_error = None
        self.mena_squared_error = None
        self.root_mean_square_error = None
        self.init_ui()

    def init_ui(self):
        """
        Second main object. There is set geometry of main window
        and it refers to main grafical elements.

        :return: None
        """
        self.label_1 = QtWidgets.QLabel(self)
        self.label = QtWidgets.QLabel(self)
        self.label_2 = QtWidgets.QLabel(self)
        self.label_3 = QtWidgets.QLabel(self)
        self.label_mae = QtWidgets.QLabel(self)
        self.label_mse = QtWidgets.QLabel(self)
        self.label_rmse = QtWidgets.QLabel(self)
        self.table_widget = QTableWidget(self)
        # self.textbox4 = QLineEdit(self)
        self.textbox3 = QLineEdit(self)
        self.textbox2 = QLineEdit(self)
        self.textbox1n = QLineEdit(self)
        self.textbox1s = QLineEdit(self)
        self.textbox1 = QLineEdit(self)
        self.button_one = QPushButton("Intr.speed detection", self)
        self.button_two = QPushButton("One speed detection", self)
        # self.button_three = QPushButton("Test detection", self)
        self.box = QCheckBox("Save output", self)
        self.check_box_one()
        self.setGeometry(100, 100, 700, 580)  # 700
        self.setWindowTitle("AND - Application for novelty detection")
        self.label_one()
        self.tabulka()
        self.menugraf()
        self.popisky()
        self.combobox()
        self.lspeed()
        self.selection_window_button()
        self.show()

    def error_evaluation(self):
        """
        Object which calculate filter error statistics
        :return: None
        """
        self.mean_absolute_error = pa.misc.MAE(self.filter_error)
        self.mena_squared_error = pa.misc.MSE(self.filter_error)
        self.root_mean_square_error = pa.misc.RMSE(self.filter_error)
        self.logarithmic_squared_error = pa.misc.logSE(self.filter_error)

    def new_selection(self):
        """
        Object which create table of input matrix size
        :return: None
        """
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(4)
        self.table_widget.setColumnCount(4)
        self.table_widget.addAction(self, QCheckBox())
        self.table_widget.show()

    def check_box_one(self):
        """
        Object which define checkbox for saving data.
        :return: None
        """
        self.box.move(560, 350)
        self.box.resize(320, 40)

    def check_box_one_save(self):
        """
        Object which make dirs for save data and save them.
        :return: None
        """
        if self.box.isChecked():
            try:
                self.directory_3 = str(self.textbox2.text())
                parent_dir_a = "C:\\Program Files (x86)\\Exp.files\\"
                self.parent_dir_3 = parent_dir_a + self.directory_2
                self.path_3 = os.path.join(self.parent_dir_3, self.directory_3)
                try:
                    os.makedirs(self.path_3)
                    print("dir maked")
                except Exception as ex:
                    print(ex)
            except Exception as ex:
                print(ex)
        else:
            pass

    def check_box_one_save_main_file(self):
        """
        Object which make dirs for save data and save them.
        :return: None
        """
        if self.box.isChecked():
            directory_1 = "Exp.files"
            parent_dir_1 = "C:\Program Files (x86)"
            path_1 = os.path.join(parent_dir_1, directory_1)
            try:
                os.makedirs(path_1)
                print("dir maked")
            except Exception as ex:
                print(ex)

            list_files = os.listdir('C:\\Program Files (x86)\\Exp.files')

            self.directory_2 = str(len(list_files))

            parent_dir_2 = "C:\\Program Files (x86)\\Exp.files"
            path_2 = os.path.join(parent_dir_2, self.directory_2)

            try:
                os.makedirs(path_2)
                print("dir maked")
            except Exception as ex:
                print(ex)

        else:
            print("not checked")

    def combobox(self):
        """
        This object add items into combobox of filters
        so we can choose filter by another object
        :return: None
        """
        self.combo_1 = QComboBox(self)
        self.combo_2 = QComboBox(self)
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
        self.combo_1.addItem("GMCC")
        self.combo_1.addItem("Llncosh")
        self.combo_1.setGeometry(550, 80, 130, 25)
        self.combo_2.addItem("Choose detection")
        self.combo_2.addItem("LE")
        self.combo_2.addItem("ELBND")
        self.combo_2.addItem("ESE")
        self.combo_2.setGeometry(550, 110, 130, 25)
        self.button_one.setGeometry(550, 495, 130, 25)
        self.button_one.clicked.connect(self.node_2)
        self.button_two.setGeometry(550, 525, 130, 25)
        self.button_two.clicked.connect(self.node_3)

    def multi_parametrs_save(self):
        """
        Object which define function for button save parametrs.
        These parametrs are saved into the .txt file.
        :return: None
        """
        try:
            if self.box.isChecked():
                spn = str(self.textbox1.text())
                name = str(self.textbox2.text())
                stdev = str(self.standart_deviation)
                variance = str(self.variance)
                mean = str(self.mean)

                mae = str(self.mean_absolute_error)
                mse = str(self.mena_squared_error)
                rmse = str(self.root_mean_square_error)

                filter_lenght_string = str(self.textbox3.text())
                saved_parametrs = {"Learning rate": spn,\
                                   "Filter length": filter_lenght_string,\
                                   "Filter": self.filter_name, \
                                   "Detection tool": self.detection_name,\
                                   "Standart deviation": stdev, \
                                   "Variance": variance,\
                                   "Mean": mean,\
                                   "Mean absolute error": mae,\
                                   "Mean squared error": mse,\
                                   "Root mean square error": rmse}
                speed_label = str(self.label_savefig)

                name_of_file = str(self.directory_3 + speed_label)
                complete_name = os.path.join(self.path_3, name_of_file + ".txt")

                saved_parametrs_string = repr(saved_parametrs)
                file_parametrs = open(complete_name,"w", encoding="utf-8")
                file_parametrs.write(saved_parametrs_string)
                file_parametrs.close()
        except Exception as ex:
            print(ex)

    def node_3(self):
        """
        Node for selection of multiple columns, when you click
        on that buton you start this node which refers
        to individual objects
        :return: None
        """
        self.check_box_one_save_main_file()
        self.main_node()

    def main_node(self):
        """
        Main structure of node which is use in specialized nodes
        :return: None
        """
        self.lspeedchoose()
        self.alter_vyber()
        if self.num_of_parametrs == 0:
            self.num_of_parametrs = 1
            self.vyberfiltr()
            self.error_evaluation()
            self.vyberdet()
            self.check_box_one_save()
            self.multi_parametrs_save()
            self.grafy()
        else:
            self.vyberfiltr()
            self.error_evaluation()
            self.vyberdet()
            self.check_box_one_save()
            self.multi_parametrs_save()
            self.grafy()

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

        self.check_box_one_save_main_file()

        for self.learning_rate in np.arange(bottom_interval, \
                                            top_interval, step):

            self.label_savefig = round(self.learning_rate, 5)
            self.main_node()

    def vyberfiltr(self):
        """
        This object choose which adaptive filter we use.
        :return: None
        """
        filt = self.combo_1.currentText()
        if filt == "SSLMS":
            self.filter_sslms()
        if filt == "RLS":
            self.filter_rls()
        elif filt == "NSSLMS":
            self.filter_nsslms()
        elif filt == "AP":
            self.filter_ap()
        elif filt == "GNGD":
            self.filter_gngd()
        elif filt == "NLMF":
            self.filter_nlmf()
        elif filt == "NLMS":
            self.filter_nlms()
        elif filt == "LMS":
            self.filter_lms()
        elif filt == "LMF":
            self.filter_lmf()
        elif filt == "GMCC":
            self.filter_gmcc()
        elif filt == "Llncosh":
            self.filter_llncosh()

    def vyberdet(self):
        """
        This object choose which detection tool we use.
        :return: None
        """
        det = self.combo_2.currentText()
        if det == "LE":
            self.det_le()
        elif det == "ELBND":
            self.det_elbnd()
        elif det == "ESE":
            self.det_ese()

    def menufil(self):
        """
        Object which define options of filter and add them into combobox.
        :return: None
        """
        self.statusBar()
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&Filters")

        llncosh = QAction(QIcon("WWW.jpg"), "llncosh", self)
        llncosh.setShortcut("Ctrl+ll")
        llncosh.setStatusTip("Last Llncosh (Llncosh)")
        llncosh.triggered.connect(self.filter_llncosh)

        gmcc = QAction(QIcon("WWW.jpg"), "GMCC", self)
        gmcc.setShortcut("Ctrl+G")
        gmcc.setStatusTip("Generalized maximum correntropy criterion (GMCC)")
        gmcc.triggered.connect(self.filter_gmcc)

        sslms = QAction(QIcon("WWW.jpg"), "SSLMS", self)
        sslms.setShortcut("Ctrl+F")
        sslms.setStatusTip("Sign-sign Least-mean-squares (SSLMS)")
        sslms.triggered.connect(self.filter_sslms)

        rls = QAction(QIcon("WWW.jpg"), "RLS", self)
        rls.setShortcut("Ctrl+R")
        rls.setStatusTip("Recursive Least Squares (RLS)")
        rls.triggered.connect(self.filter_rls)

        nsslms = QAction(QIcon("WWW.jpg"), "NSSLMS", self)
        nsslms.setShortcut("Ctrl+Q")
        nsslms.setStatusTip("Normalized Sign-sign Least-mean-squares (NSSLMS)")
        nsslms.triggered.connect(self.filter_nsslms)

        ap_filter = QAction(QIcon("WWW.jpg"), "AP", self)
        ap_filter.setShortcut("Ctrl+A")
        ap_filter.setStatusTip("Affine Projection (AP)")
        ap_filter.triggered.connect(self.filter_ap)

        gngd = QAction(QIcon("WWW.jpg"), "GNGD", self)
        gngd.setShortcut("Ctrl+G")
        gngd.setStatusTip("Generalized Normalized Gradient Descent (GNGD)")
        gngd.triggered.connect(self.filter_gngd)

        nlms = QAction(QIcon("WWW.jpg"), "NLMS", self)
        nlms.setShortcut("Ctrl+N")
        nlms.setStatusTip("Normalized Least-mean-squares (NLMS)")
        nlms.triggered.connect(self.filter_nlms)

        nlmf = QAction(QIcon("WWW.jpg"), "NLMF", self)
        nlmf.setShortcut("Ctrl+F")
        nlmf.setStatusTip("Normalized Least-mean-fourth (NLMF)")
        nlmf.triggered.connect(self.filter_nlmf)

        lms = QAction(QIcon("WWW.jpg"), "LMS", self)
        lms.setShortcut("Ctrl+S")
        lms.setStatusTip("Least-mean-fourth (LMS)")
        lms.triggered.connect(self.filter_lms)

        lmf = QAction(QIcon("WWW.jpg"), "LMF", self)
        lmf.setShortcut("Ctrl+M")
        lmf.setStatusTip("Least-mean-fourth (LMF)")
        lmf.triggered.connect(self.filter_lmf)

        file_menu.addAction(gmcc)
        file_menu.addAction(sslms)
        file_menu.addAction(rls)
        file_menu.addAction(ap_filter)
        file_menu.addAction(nsslms)
        file_menu.addAction(gngd)
        file_menu.addAction(nlms)
        file_menu.addAction(nlmf)
        file_menu.addAction(lms)
        file_menu.addAction(lmf)
        file_menu.addAction(llncosh)

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
        elbnd_detection.triggered.connect(self.det_elbnd)

        le_detection = QAction(QIcon("WWW.jpg"), "LE", self)
        le_detection.setShortcut("Ctrl+2")
        le_detection.setStatusTip("Learning Entropy (LE)")
        le_detection.triggered.connect(self.det_le)

        ese_detection = QAction(QIcon("WWW.jpg"), "ESE", self)
        ese_detection.setShortcut("Ctrl+3")
        ese_detection.setStatusTip("Extreme seeking entrophy (ESE)")
        ese_detection.triggered.connect(self.det_ese)

        file_menu.addAction(elbnd_detection)
        file_menu.addAction(le_detection)
        file_menu.addAction(ese_detection)

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
        self.setGeometry(QtCore.QRect(550, 40, 250, 710))
        self.setGeometry(100, 100, 700, 580)

        variance_multiplied = 1000 * self.variance
        variance_back = (round(variance_multiplied)) / 1000
        variance_round = str(variance_back)
        self.label_1.setText("Var:" + variance_round)
        self.label_1.setStyleSheet("font-weight:bold")
        self.label_1.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(550, 40, 250, 715))
        self.setGeometry(100, 100, 700, 580)

        deviation_multiplied = 1000 * self.standart_deviation
        deviation_back = (round(deviation_multiplied)) / 1000
        deviation_round = str(deviation_back)
        self.label_2.setText("Std:" + deviation_round)
        self.label_2.setStyleSheet("font-weight:bold")
        self.label_2.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(550, 40, 250, 704))
        self.setGeometry(100, 100, 700, 580)

        mean_absolute_error_multiplied = 1000 * self.mean_absolute_error
        mean_absolute_error_back = (round(mean_absolute_error_multiplied)) / 1000
        mean_absolute_error_round = str(mean_absolute_error_back)
        self.label_mae.setText("Std:" +  mean_absolute_error_round)
        self.label_mae.setStyleSheet("font-weight:bold")
        self.label_mae.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(550, 40, 250, 740))
        self.setGeometry(100, 100, 700, 580)

        mena_squared_error_multiplied = 1000 * self.mena_squared_error
        mena_squared_error_back = (round(mena_squared_error_multiplied)) / 1000
        mena_squared_error_round = str(mena_squared_error_back)
        self.label_mse.setText("Std:" + mena_squared_error_round)
        self.label_mse.setStyleSheet("font-weight:bold")
        self.label_mse.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(550, 40, 250, 775))
        self.setGeometry(100, 100, 700, 580)

        root_mean_square_error_multiplied = 1000 * self.root_mean_square_error
        root_mean_square_error_back = (round(root_mean_square_error_multiplied)) / 1000
        root_mean_square_error_round = str(root_mean_square_error_back)
        self.label_rmse.setText("Std:" + root_mean_square_error_round)
        self.label_rmse.setStyleSheet("font-weight:bold")
        self.label_rmse.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(550, 40, 250, 810))
        self.setGeometry(100, 100, 700, 580)


    def statokno(self):
        """
        Object which calculate statistic.
        :return: None
        """
        stlist = np.ndarray.tolist(self.output_detection_tool)
        self.mean = statistics.mean(stlist)
        self.standart_deviation = statistics.stdev(stlist)
        self.variance = statistics.variance(stlist)

    def label_one(self):
        """
        Object which define statistic labels for the first time.
        :return: None
        """
        self.label.setText("Mean: None")
        self.label.setStyleSheet("font-weight:bold")
        self.label.setStyleSheet("font-size: 10pt")
        self.label.setGeometry(QtCore.QRect(550, 40, 250, 715))
        self.setGeometry(100, 100, 700, 700)
        self.label.update()

        self.label_1.setText("Var: None")
        self.label_1.setStyleSheet("font-weight:bold")
        self.label_1.setStyleSheet("font-size: 10pt")
        self.label_1.setGeometry(QtCore.QRect(550, 60, 250, 710))
        self.setGeometry(100, 100, 700, 580)
        self.label_1.update()

        self.label_2.setText("Std: None")
        self.label_2.setStyleSheet("font-weight:bold")
        self.label_2.setStyleSheet("font-size: 10pt")
        self.label_2.setGeometry(QtCore.QRect(550, 80, 250, 704))
        self.setGeometry(100, 100, 700, 580)
        self.label_2.update()

        self.label_mae.setText("Mae: None")
        self.label_mae.setStyleSheet("font-weight:bold")
        self.label_mae.setStyleSheet("font-size: 10pt")
        self.label_mae.setGeometry(QtCore.QRect(550, 80, 250, 740))
        self.setGeometry(100, 100, 700, 580)
        self.label_mae.update()

        self.label_mse.setText("Mse: None")
        self.label_mse.setStyleSheet("font-weight:bold")
        self.label_mse.setStyleSheet("font-size: 10pt")
        self.label_mse.setGeometry(QtCore.QRect(550, 80, 250, 775))
        self.setGeometry(100, 100, 700, 580)
        self.label_mse.update()

        self.label_rmse.setText("Rmse: None")
        self.label_rmse.setStyleSheet("font-weight:bold")
        self.label_rmse.setStyleSheet("font-size: 10pt")
        self.label_rmse.setGeometry(QtCore.QRect(550, 80, 250, 810))
        self.setGeometry(100, 100, 700, 580)
        self.label_rmse.update()

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
        """
        In that object is open file with data and reorganize into matrix for analyze.
        :return: None
        """
        try:
            basefile = str(Path.home())
            jmeno = QFileDialog.getOpenFileName(self, "open file", basefile)
            # hodnoty = open(jmeno[0], "r")
            with open(jmeno[0], "r", encoding="utf-8") as hodnoty:
                mat1 =(np.genfromtxt(hodnoty, delimiter=",", skip_header=0))
            loading_matrix_shape = mat1.shape
            self.loading_matrix_shape = mat1.shape[0]
            self.loading_matrix = mat1
            try:
                self.column_count = loading_matrix_shape[1]
            except:
                self.column_count = 1
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
        except Exception as ex:
            print(ex)

    def alter_vyber(self):
        count = self.selected_columns
        self.num_of_parametrs = len(count[0])
        if self.num_of_parametrs == 0:
            try:
                self.operational_matrix = []
                self.input_columns_filter = []
                self.desired_columns_filter = []
                loading_matrix = self.loading_matrix
                lenght_of_data = self.loading_matrix_shape
                for i in range(0, lenght_of_data):
                    self.operational_matrix.append(loading_matrix[i, count[1][0]])
                for i in range(0, lenght_of_data-5):
                    self.input_columns_filter.append(self.operational_matrix[i])
                for i in range(5, lenght_of_data):
                    self.desired_columns_filter.append(self.operational_matrix[i])
                self.desired_columns_filter = np.reshape(self.desired_columns_filter,\
                                                         [lenght_of_data-5, len(count[0])+1],
                                                         order="F")
                self.input_data = np.asarray(self.input_columns_filter)
                self.input_desired_data = np.asarray(self.desired_columns_filter)
            except Exception as ex:
                print(ex)
                for i in range(0, lenght_of_data):
                    self.operational_matrix.append(loading_matrix[i])
                for i in range(0, lenght_of_data - 5):
                    self.input_columns_filter.append(self.operational_matrix[i])
                for i in range(5, lenght_of_data):
                    self.desired_columns_filter.append(self.operational_matrix[i])
                self.desired_columns_filter = np.reshape(self.desired_columns_filter,
                                                         [lenght_of_data - 5, len(count[0]) + 1],
                                                         order="F")
                self.input_data = np.asarray(self.input_columns_filter)
                self.input_desired_data = np.asarray(self.desired_columns_filter)
        else:
            self.input_columns_filter = []
            self.desired_columns_filter = []
            loading_matrix = self.loading_matrix
            lenght_of_data = self.loading_matrix_shape
            for i in range(0, lenght_of_data):
                self.input_columns_filter.append(loading_matrix[i, count[1][0]])
            for j in range(0, len(count[0])):
                for i in range(0, lenght_of_data):
                    self.desired_columns_filter.append(loading_matrix[i, count[0][j]])
            self.desired_columns_filter = np.reshape(self.desired_columns_filter,\
                                                     [lenght_of_data, len(count[0])],
                                                     order="F")
            self.input_data = np.asarray(self.input_columns_filter)
            self.input_desired_data = np.asarray(self.desired_columns_filter)

    def vyber(self):
        """
        Function of this object is selection of column marked
        for analyzation and insert this column into the matrix.
        :return: None
        """
        alt = self.table_widget.selectedItems()
        selection_list = []
        selection_list_shape = self.matrix_shape[0]

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
        """
        Object which reorganize data from selected column into two matrixes.
        First matrix is with input values and second is
        filed with desired values.
        :return: None
        """
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

            print(self.input_desired_data)
            print(self.input_data)

        except Exception as ex:
            print(ex)
            trans_matrix = self.transposed_matrix
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

        print(self.input_desired_data)
        print(self.input_data)

    def lspeed(self):
        """
        Objcet which define textboxes.
        :return: None
        """
        self.textbox1.setGeometry(550, 250, 30, 25)
        self.textbox1s.setGeometry(600, 250, 30, 25)
        self.textbox1n.setGeometry(650, 250, 30, 25)
        self.textbox2.setGeometry(550, 300, 130, 25)
        self.textbox3.setGeometry(550, 200, 130, 25)
        self.show()

    def saveparametrs(self):
        """
        Object which define function for button save parametrs.
        These parametrs are saved into the .txt file.
        :return: None
        """
        try:
            spn = str(self.textbox1.text())
            name = str(self.textbox2.text())
            stdev = str(self.standart_deviation)
            variance = str(self.variance)
            mean = str(self.mean)
            filter_lenght_string = str(self.textbox3.text())
            saved_parametrs = {"Learning rate": spn, "Filter length": \
                filter_lenght_string, "Filter": self.filter_name, \
                               "Detection tool": self.detection_name, "Standart deviation": stdev, \
                               "Variance": variance, "Mean": mean}
            saved_parametrs_string=repr(saved_parametrs)
            write_parametrs = open(f"{name}.txt", "w+")
            write_parametrs.write(saved_parametrs_string)
            self.statusBar().showMessage("Parametrs save")
            write_parametrs.close()
        except Exception as ex:
            print(ex)
            saved_parametrs = {"Learning rate": self.learning_rate, \
                               "Filter length": filter_lenght_string, \
                               "Filter": self.filter_name,"Detection tool": self.detection_name, \
                               "Standart deviation": stdev, \
                               "Variance": variance, "Mean": mean}
            saved_parametrs_string = repr(saved_parametrs)
            #write_parametrs = open("Detection parametrs", "w+")
            with open("Detection parametrs", "w+", encoding="utf-8") as write_parametrs:
                write_parametrs.write(saved_parametrs_string)
            write_parametrs.close()

    def lspeedchoose(self):
        """
        Object which set learning speed written into the textbox.
        If textbox is empty, learning rate is set to 1.
        :return: None
        """
        try:
            speedvalue = float(self.textbox1.text())
            self.learning_rate = speedvalue
        except Exception as ex:
            print(ex)
            self.learning_rate = 1

    def filter_llncosh(self):
        """
               Filter procesing and values assignment
               :return: None
               """
        self.filter_name = "Llncosh"
        try:
            proces_file = pa.filters.FilterLlncosh(n=self.num_of_parametrs,\
                                                   mu=self.learning_rate,lambd=0.1, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("Llncosh filter aplicated")

    def filter_gmcc(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "GMCC"
        try:
            proces_file = pa.filters.FilterGMCC(n=self.num_of_parametrs,\
                                                mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("GMCC filter aplicated")

    def filter_sslms(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "SSLMS"
        try:
            proces_file = pa.filters.FilterSSLMS(n=self.num_of_parametrs,\
                                                 mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("SSLMS filter aplicated")

    def filter_rls(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "RLS"
        try:
            proces_file = pa.filters.FilterRLS(n=self.num_of_parametrs,\
                                               mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("RLS filter aplicated")

    def filter_nsslms(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "NSSLMS"
        try:
            proces_file = pa.filters.FilterNSSLMS(n=self.num_of_parametrs,\
                                                  mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("NSSLMS filter aplicated")

    def filter_ap(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "AP"
        try:
            proces_file = pa.filters.FilterAP(n=self.num_of_parametrs,\
                                              mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("AP filter aplicated")

    def filter_nlms(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "NLMS"
        try:
            proces_file = pa.filters.FilterNLMS(n=self.num_of_parametrs,\
                                                mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("NLMS filter aplicated")

    def filter_gngd(self):                # mam prohozene vstupy a výstupy...
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "GNGD"
        try:
            proces_file = pa.filters.FilterGNGD(n=self.num_of_parametrs,\
                                                mu=self.learning_rate, w="random")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("GNGD filter aplicated")

    def filter_nlmf(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "NLMF"
        try:
            proces_file = pa.filters.FilterNLMF(n=self.num_of_parametrs,\
                                                mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("NLMF filter aplicated")

    def filter_lms(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "LMS"
        try:
            proces_file = pa.filters.FilterLMS(n=self.num_of_parametrs,\
                                               mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("LMS filter aplicated")

    def filter_lmf(self):
        """
        Filter procesing and values assignment
        :return: None
        """
        self.filter_name = "LMF"
        try:
            proces_file = pa.filters.FilterLMF(n=self.num_of_parametrs,\
                                               mu=self.learning_rate, w="zeros")
            output, error, weights = proces_file.run(self.input_data, self.input_desired_data)
        except Exception as ex:
            print(ex)
        self.filter_output = output
        self.filter_parametrs = weights
        self.filter_error = error
        self.statusBar().showMessage("LMF filter aplicated")

    def det_elbnd(self):
        """
        Detection tool procesing and values assignment
        :return: None
        """
        try:
            self.detection_name = "ELBND"
            elbnd = pa.detection.ELBND(self.filter_parametrs, self.filter_error, function="max")
            self.output_detection_tool = elbnd
            self.statusBar().showMessage("ELBND detection aplicated")
            self.statokno()
            self.statupgr()
        except Exception as ex:
            print(ex)

    def det_le(self):
        """
        Detection tool procesing and values assignment
        :return: None
        """
        try:
            self.detection_name = "LE"
            print(self.filter_parametrs)
            le_detection = pa.detection.learning_entropy(self.filter_parametrs, m=30, order=2)
            det_le_matrix_shape = le_detection.shape
            print(det_le_matrix_shape)
            print(le_detection)
            reshaped_output_matrix = np.reshape(le_detection, (det_le_matrix_shape[0],))
            self.output_detection_tool = reshaped_output_matrix
            self.statusBar().showMessage("LE detection aplicated")
            self.statokno()
            self.statupgr()
        except Exception as ex:
            print(ex)

    def det_ese(self):
        """
        Detection tool procesing and values assignment
        :return: None
        """
        try:
            self.detection_name = "ESE"
            print(self.filter_parametrs)
            ese = pa.detection.ESE(self.filter_parametrs)
            self.output_detection_tool = ese
            self.statusBar().showMessage("ELBND detection aplicated")
            self.statokno()
            self.statupgr()
        except Exception as ex:
            print(ex)

    def tabulka(self):
        """
        Object which define table for data from .csv file.
        :return: None
        """
        self.table_widget.setRowCount(20)
        self.table_widget.setColumnCount(5)
        self.table_widget.setGeometry(25, 50, 500, 500)

    def grafy(self):
        """
        Object which define graphs for output from filter
        and output from detection tool.
        :return: None
        """
        fig, axs = plt.subplots(4, 1)
        axs[0].plot(self.filter_error)
        axs[1].plot(self.filter_output)
        axs[2].plot(self.filter_parametrs)
        axs[3].plot(self.output_detection_tool)

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

        try:
            origin_directory = os.getcwd()
            parent_dir_4 = self.parent_dir_3 + "\\" + self.directory_3
            os.chdir(parent_dir_4)
            current_setup = "R" + str(self.textbox2.text()) + str(self.label_savefig) + ".jpg"
            plt.savefig(current_setup)
            os.chdir(origin_directory)
        except Exception as ex:
            print(ex)

    def selection_window_button(self):
        """
        Object inicializing second window and emiting signal of input data to it.
        :return: None
        """
        # Button
        self.button = QPushButton(self)
        self.button.setGeometry(550, 140, 130, 25)
        self.button.setText('Selection')
        self.button.setStyleSheet('font-size:15px')
        self.button.show()

        # Button Event
        self.button.clicked.connect(self.column_number_list_update)

    def column_number_list_update(self):
        """
        Object emiting signal of input data to second window.
        :return: None
        """
        # variable passing
        self.selection_window.column_number = self.column_count
        self.selection_window.checkboxes()
        self.selection_window.column_selection_button()
        self.selection_window.show()

    def column_selection_holder_one(self, data_one: list):
        self.selected_columns = []
        self.selected_columns = data_one

    def closeEvent(self, event):
        """
        CloseEvent is function which questioned you if you
        realy want close app.
        :return: None
        """
        reply = QMessageBox.question(self, 'Window Close', \
                                     'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class SubWindow(QMainWindow):
    signal_proces = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.setGeometry(1060, 100, 270, 580)
        self.setWindowTitle("Selection")

        # attribute definition
        self.scroll = QScrollArea()
        self.widget = QWidget()
        self.vbox = QVBoxLayout()

        # global variables and lists
        self.column_number = None
        self.matrix_of_column_selection = []
        self.in_checkboxes = []
        self.out_checkboxes = []
        self.in_column_numbers = []
        self.out_column_numbers = []

        # other methods

    def scroll_setup(self):
        # scroll area properties
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.widget)
        self.setCentralWidget(self.scroll)

    def checkboxes(self):
        self.scroll_setup()

        for i in range(0, self.column_number):
            name = "target-col " + str(i)
            checkbox_in = QCheckBox(name)
            self.in_checkboxes.append(checkbox_in)
            self.vbox.addWidget(checkbox_in)
        self.widget.setLayout(self.vbox)

        for i in range(0, self.column_number):
            name = "in-col " + str(i)
            checkbox_out = QCheckBox(name)
            self.out_checkboxes.append(checkbox_out)
            self.vbox.addWidget(checkbox_out)
        self.widget.setLayout(self.vbox)

    def column_selection_button(self):
        self.selection_button = QPushButton("Selection")
        self.selection_button.setGeometry(1070, 600, 20, 20)
        self.vbox.addWidget(self.selection_button)
        self.selection_button.show()
        self.selection_button.clicked.connect(self.column_selection)
        self.selection_button.clicked.connect(self.signal_emit_back)

    def column_selection(self):
        self.matrix_of_column_selection = []
        self.out_column_numbers = []
        self.in_column_numbers = []
        for i, checkbox in enumerate(self.in_checkboxes):
            if checkbox.isChecked():
                self.in_column_numbers.append(i)
        for i, checkbox in enumerate(self.out_checkboxes):
            if checkbox.isChecked():
                self.out_column_numbers.append(i)
        self.matrix_of_column_selection = [self.out_column_numbers, self.in_column_numbers]

    def signal_emit_back(self):
        self.signal_proces.emit(self.matrix_of_column_selection)

def main():
    """
    Main section, system exit from application.
    :return: None
    """
    app = QApplication(sys.argv)
    ex_ex = Detekce()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()
