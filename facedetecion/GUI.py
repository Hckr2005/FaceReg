# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt6 UI code generator 6.7.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_facedetection(object):
    def setupUi(self, facedetection):
        facedetection.setObjectName("facedetection")
        facedetection.resize(905, 673)
        self.centralwidget = QtWidgets.QWidget(parent=facedetection)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(parent=self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 10, 261, 111))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setMouseTracking(False)
        self.label.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(parent=self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(90, 150, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.pushButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(490, 140, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(parent=self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(180, 150, 241, 31))
        self.textEdit.setObjectName("textEdit")
        facedetection.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(parent=facedetection)
        self.statusbar.setObjectName("statusbar")
        facedetection.setStatusBar(self.statusbar)

        self.retranslateUi(facedetection)
        QtCore.QMetaObject.connectSlotsByName(facedetection)

    def retranslateUi(self, facedetection):
        _translate = QtCore.QCoreApplication.translate
        facedetection.setWindowTitle(_translate("facedetection", "Dm tim mai ko ra"))
        self.label.setText(_translate("facedetection", "Welcome"))
        self.label_2.setText(_translate("facedetection", "nhap"))
        self.pushButton.setText(_translate("facedetection", "OK"))
        self.textEdit.setHtml(_translate("facedetection", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    facedetection = QtWidgets.QMainWindow()
    ui = Ui_facedetection()
    ui.setupUi(facedetection)
    facedetection.show()
    sys.exit(app.exec())