"""
GUI for adaptive filters testing.
"""
import sys
from pathlib import Path

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtGui import *
import numpy as np
import matplotlib.pyplot as plt
import padasip as pa
import statistics

# pylint: disable=too-many-instance-attributes
# pylint: disable=c-extension-no-member

class Detekce(QMainWindow):
    """
    Popis
    """
    def __init__(self):
        QMainWindow.__init__(self)

        self.m = None
        self.q = None

        self.y = None
        self.w = None
        self.e = None
        self.d = None
        self.x = None
        self.det = None
        self.rl = None
        self.nmf = None
        self.nmd = None

        self.mean = None
        self.STDEV = None
        self.VARIANCE = None

        self.initUI()

    def initUI(self):
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
        self.comboI = QComboBox(self)
        self.comboI.addItem("Choose filter")
        self.comboI.addItem("SSLMS")
        self.comboI.addItem("RLS")
        self.comboI.addItem("NSSLMS")
        self.comboI.addItem("AP")
        self.comboI.addItem("GNGD")
        self.comboI.addItem("NLMF")
        self.comboI.addItem("NLMS")
        self.comboI.addItem("LMS")
        self.comboI.addItem("LMF")
        self.comboI.setGeometry(550, 80, 130, 25)

        self.comboII = QComboBox(self)
        self.comboII.addItem("Choose detection")
        self.comboII.addItem("LE")
        self.comboII.addItem("ELBND")
        self.comboII.setGeometry(550, 110, 130, 25)

        self.spoust = QPushButton("Intr.speed detection", self)
        self.spoust.setGeometry(550, 495, 130, 25)
        self.spoust.clicked.connect(self.uzelII)

        self.spoust = QPushButton("One speed detection", self)
        self.spoust.setGeometry(550, 525, 130, 25)
        self.spoust.clicked.connect(self.uzel)

    def uzelII(self):
        h1 = int(self.textbox1.text())
        h2 = int(self.textbox1s.text())
        h3 = int(self.textbox1n.text())
        for self.rl in range(h1,h2,h3):
            self.vyber()
            self.vyberfiltr()
            self.vyberdet()
            self.grafy()

    def uzel(self):
        """
        Docstring
        :return: None
        """
        self.lspeedchoose()
        self.vyber()
        self.vyberfiltr()
        self.vyberdet()
        self.grafy()

    def vyberfiltr(self):
        filt = self.comboI.currentText()
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
        det = self.comboII.currentText()
        if det == "LE":
            self.detLE()
        elif det == "ELBND":
            self.detELBND()

    def menufil(self):
        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&Filters")

        SSLMS = QAction(QIcon("WWW.jpg"), "SSLMS", self)
        SSLMS.setShortcut("Ctrl+F")
        SSLMS.setStatusTip("Sign-sign Least-mean-squares (SSLMS)")
        SSLMS.triggered.connect(self.filterSSLMS)

        RLS = QAction(QIcon("WWW.jpg"), "RLS", self)
        RLS.setShortcut("Ctrl+R")
        RLS.setStatusTip("Recursive Least Squares (RLS)")
        RLS.triggered.connect(self.filterRLS)

        NSSLMS = QAction(QIcon("WWW.jpg"), "NSSLMS", self)
        NSSLMS.setShortcut("Ctrl+Q")
        NSSLMS.setStatusTip("Normalized Sign-sign Least-mean-squares (NSSLMS)")
        NSSLMS.triggered.connect(self.filterNSSLMS)

        AP = QAction(QIcon("WWW.jpg"), "AP", self)
        AP.setShortcut("Ctrl+A")
        AP.setStatusTip("Affine Projection (AP)")
        AP.triggered.connect(self.filterAP)

        GNGD = QAction(QIcon("WWW.jpg"), "GNGD", self)
        GNGD.setShortcut("Ctrl+G")
        GNGD.setStatusTip("Generalized Normalized Gradient Descent (GNGD)")
        GNGD.triggered.connect(self.filterGNGD)

        NLMS = QAction(QIcon("WWW.jpg"), "NLMS", self)
        NLMS.setShortcut("Ctrl+N")
        NLMS.setStatusTip("Normalized Least-mean-squares (NLMS)")
        NLMS.triggered.connect(self.filterNLMS)

        NLMF = QAction(QIcon("WWW.jpg"), "NLMF", self)
        NLMF.setShortcut("Ctrl+F")
        NLMF.setStatusTip("Normalized Least-mean-fourth (NLMF)")
        NLMF.triggered.connect(self.filterNLMF)

        LMS = QAction(QIcon("WWW.jpg"), "LMS", self)
        LMS.setShortcut("Ctrl+S")
        LMS.setStatusTip("Least-mean-fourth (LMS)")
        LMS.triggered.connect(self.filterLMS)

        LMF = QAction(QIcon("WWW.jpg"), "LMF", self)
        LMF.setShortcut("Ctrl+M")
        LMF.setStatusTip("Least-mean-fourth (LMF)")
        LMF.triggered.connect(self.filterLMF)

        fileMenu.addAction(SSLMS)
        fileMenu.addAction(RLS)
        fileMenu.addAction(AP)
        fileMenu.addAction(NSSLMS)
        fileMenu.addAction(GNGD)
        fileMenu.addAction(NLMS)
        fileMenu.addAction(NLMF)
        fileMenu.addAction(LMS)
        fileMenu.addAction(LMF)

    def menuDET(self):
        self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&Detection")

        ELBND = QAction(QIcon("WWW.jpg"), "ELBND", self)
        ELBND.setShortcut("Ctrl+1")
        ELBND.setStatusTip("Error and Learning Based Novelty Detection (ELBND)")
        ELBND.triggered.connect(self.detELBND)

        LE = QAction(QIcon("WWW.jpg"), "LE", self)
        LE.setShortcut("Ctrl+2")
        LE.setStatusTip("Learning Entropy (LE)")
        LE.triggered.connect(self.detLE)

        fileMenu.addAction(ELBND)
        fileMenu.addAction(LE)

    def popisky(self):
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
        ant = 1000 * self.mean
        art = (round(ant))/1000
        ART = str(art)
        self.label.setText("Mean:" + ART)
        self.label.setStyleSheet("font-weight:bold")
        self.label.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)

        ans = 1000 * self.VARIANCE
        ars = (round(ans)) / 1000
        ARS = str(ars)
        self.label_1.setText("Variance:" + ARS)
        self.label_1.setStyleSheet("font-weight:bold")
        self.label_1.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)

        anp = 1000 * self.STDEV
        arp = (round(anp)) / 1000
        ARP = str(arp)
        self.label_2.setText("Standart deviation:" + ARP)
        self.label_2.setStyleSheet("font-weight:bold")
        self.label_2.setStyleSheet("font-size: 10pt")
        self.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)

    def statokno(self):
        stlist = np.ndarray.tolist(self.det)
        self.mean = statistics.mean(stlist)
        self.STDEV = statistics.stdev(stlist)
        self.VARIANCE = statistics.variance(stlist)
        print(self.mean)
        print(self.STDEV)
        print(self.VARIANCE)


    def label_I(self):
        self.label =QtWidgets.QLabel(self)
        self.label.setText("Mean: None")
        self.label.setStyleSheet("font-weight:bold")
        self.label.setStyleSheet("font-size: 10pt")
        self.label.setGeometry(QtCore.QRect(700, 40, 250, 50))
        self.setGeometry(100, 100, 900, 580)
        self.label.update()

        self.label_1 = QtWidgets.QLabel(self)
        self.label_1.setText("Variance: None")
        self.label_1.setStyleSheet("font-weight:bold")
        self.label_1.setStyleSheet("font-size: 10pt")
        self.label_1.setGeometry(QtCore.QRect(700, 60, 250, 50))
        self.setGeometry(100, 100, 900, 580)
        self.label_1.update()

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setText("Standart deviation: None")
        self.label_2.setStyleSheet("font-weight:bold")
        self.label_2.setStyleSheet("font-size: 10pt")
        self.label_2.setGeometry(QtCore.QRect(700, 80, 250, 50))
        self.setGeometry(100, 100, 900, 580)
        self.label_2.update()

    def menugraf(self):
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

        vyb = QAction(QIcon("WWW.jpg"), "Selection", self)
        vyb.setShortcut("Ctrl+6")
        vyb.setStatusTip("Selected a column")
        vyb.triggered.connect(self.vyber)

        # self.statusBar()
        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&Tools")
        fileMenu.addAction(opfil)

    def loading(self):
        basefile = str(Path.home())
        jmeno = QFileDialog.getOpenFileName(self, "open file", basefile)
        hodnoty = open(jmeno[0], "r")
        with open(jmeno[0], "r", encoding="utf-8") as hodnoty:
            mat1 =(np.genfromtxt(hodnoty, delimiter=",", skip_header=0))

        n = mat1.shape
        if len(n) == 1:
            m = (n[0], 1)
        else:
            m = n
        self.tableWidget.setRowCount(m[0])
        self.tableWidget.setColumnCount(m[1])
        for i in range(0, m[0]):
            for j in range(0, m[1]):
                try:
                    if len(n) == 1:
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(mat1[i])))
                    else:
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(mat1[i, j])))
                except Exception as ex:
                    print(ex)
        self.m = m
        self.statusBar().showMessage("Files are loaded")

    def vyber(self):
        alt = self.tableWidget.selectedItems()
        u = []
        n = self.m[0]
        print(n)

        for item in alt:
            u.append(float(item.text()))
        U = np.asarray(u)
        x = U.shape[0]
        f = x / n
        try:
            p = np.reshape(U, (int(f), n))
            q = p.T
            self.q = q
        except Exception as ex:
            print(ex)
        self.statusBar().showMessage("Input column are selected")

    def uprava(self):
        try:
            s = self.q
            n = self.m[0]

            sr = int(self.textbox3.text())

            x = []
            d = []

            for i in range(sr, n, 1):
                d = np.append(d, s[i])
            for i in range(0, n - sr, 1):
                x = np.append(x, s[i])
            l = np.asarray([x])
            f = np.asarray([d])
            self.x = l.T
            self.d = f.T

        except:
            s = self.q
            print(s)
            n = self.m[0]
            x = []
            d = []

            for i in range(5, n, 1):
                d = np.append(d, s[i])
            for i in range(0, n - 5, 1):
                x = np.append(x, s[i])
            l = np.asarray([x])
            f = np.asarray([d])
            self.x = l.T
            self.d = f.T

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
            stdev = str(self.STDEV)
            variance = str(self.VARIANCE)
            mean = str(self.mean)
            sr = str(self.textbox3.text())
            art = {"Learning rate": spn, "Filter length": sr, "Filter": self.nmf, "Detection tool": self.nmd, "Standart deviation": stdev, "Variance": variance, "Mean": mean}
            tra=repr(art)
            f = open("%s.txt" % name, "w+")
            f.write(tra)
            self.statusBar().showMessage("Parametrs save")
            f.close()
        except:
            art = {"Learning rate": self.rl,  "Filter length": sr, "Filter": self.nmf, "Detection tool": self.nmd, "Standart deviation": stdev, "Variance": variance, "Mean": mean}
            tra = repr(art)
            f = open("Detection parametrs","w+")
            f.write(tra)
            f.close()

    def lspeedchoose(self):
        try:
            speedvalue = float(self.textbox1.text())
            self.rl = speedvalue
        except:
            print("ok")

    def filterSSLMS(self):
        self.uprava()
        self.nmf = "SSLMS"
        try:
            f = pa.filters.FilterSSLMS(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("SSLMS filter aplicated")

    def filterRLS(self):
        self.uprava()
        self.nmf = "RLS"
        try:
            f = pa.filters.FilterRLS(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("RLS filter aplicated")

    def filterNSSLMS(self):
        self.uprava()
        self.nmf = "NSSLMS"
        try:
            f = pa.filters.FilterNSSLMS(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("NSSLMS filter aplicated")

    def filterAP(self):
        self.uprava()
        self.nmf = "AP"
        try:
            f = pa.filters.FilterAP(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("AP filter aplicated")

    def filterNLMS(self):
        self.uprava()
        self.nmf = "NLMS"
        try:
            f = pa.filters.FilterNLMS(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("NLMS filter aplicated")

    def filterGNGD(self):
        self.uprava()
        self.nmf = "GNGD"
        try:
            f = pa.filters.FilterGNGD(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("GNGD filter aplicated")

    def filterNLMF(self):
        self.uprava()
        self.nmf = "NLMF"
        try:
            f = pa.filters.FilterNLMF(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("NLMF filter aplicated")

    def filterLMS(self):
        self.uprava()
        self.nmf = "LMS"
        try:
            f = pa.filters.FilterLMS(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("LMS filter aplicated")

    def filterLMF(self):
        self.uprava()
        self.nmf = "LMF"
        try:
            f = pa.filters.FilterLMF(n=1, mu=self.rl, w="zeros")
            y, e, w = f.run(self.d, self.x)
        except Exception as ex:
            print(ex)
        self.y = y
        self.w = w
        self.e = e
        self.statusBar().showMessage("LMF filter aplicated")

    def detELBND(self):
        self.nmd = "ELBND"
        elbnd = pa.detection.ELBND(self.w, self.e, function="max")
        self.det = elbnd
        print(elbnd)
        self.statusBar().showMessage("ELBND detection aplicated")
        self.statokno()
        self.statupgr()

    def detLE(self):
        self.nmd = "LE"
        le = pa.detection.learning_entropy(self.w, m=30, order=1)
        n = le.shape
        LE = np.reshape(le, (n[0],))
        print(le)
        print(LE)
        self.det = LE
        self.statusBar().showMessage("LE detection aplicated")
        self.statokno()
        self.statupgr()

    def tabulka(self):
        self.tableWidget = QTableWidget(self)
        self.tableWidget.setRowCount(20)
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setGeometry(25, 50, 500, 500)

    def grafy(self):

        fig, axs = plt.subplots(4, 1)
        axs[0].plot(self.e)
        axs[1].plot(self.y)
        axs[2].plot(self.w)
        axs[3].plot(self.det)

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
        reply = QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()
            pass

def main():
    app = QApplication(sys.argv)
    ex = Detekce()
    sys.exit(app.exec_())

if __name__=="__main__":
    main()


