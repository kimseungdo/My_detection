# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mygui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!

import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(874, 625)
        Dialog.setStatusTip("")
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)
        self.label_4 = QtWidgets.QLabel(Dialog)
        self.label_4.setGeometry(QtCore.QRect(20, 100, 401, 501))
        self.label_4.setStyleSheet("background-color: rgb(236, 187, 255);")
        self.label_4.setText("")
        self.label_4.setObjectName("label_4") #영상이 나올 화면
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setGeometry(QtCore.QRect(440, 100, 411, 501))
        self.frame.setStyleSheet("background-color: rgb(170, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame") #결과화면
        self.textBrowser = QtWidgets.QTextBrowser(self.frame)
        self.textBrowser.setGeometry(QtCore.QRect(0, 10, 411, 481))
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Raised)
        self.textBrowser.setObjectName("textBrowser") #줄단위 결과창

        '''버튼 아이콘 생성자'''
        self.Cam_button = QtWidgets.QPushButton(Dialog)
        self.Cam_button.setGeometry(QtCore.QRect(20, 10, 71, 61))
        self.Cam_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Cam_button.setText("")
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Cam_button.setIcon(icon)
        self.Cam_button.setIconSize(QtCore.QSize(50, 50))
        self.Cam_button.setObjectName("Cam_button")
        self.Cam_button.clicked.connect(self.Cam_button_clicked) #카메라 버튼이벤트 생성
        
        self.VIdeo_button = QtWidgets.QPushButton(Dialog)
        self.VIdeo_button.setGeometry(QtCore.QRect(100, 10, 71, 61))
        self.VIdeo_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.VIdeo_button.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/play-button.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.VIdeo_button.setIcon(icon1)
        self.VIdeo_button.setIconSize(QtCore.QSize(50, 50))
        self.VIdeo_button.setObjectName("VIdeo_button")
        self.VIdeo_button.clicked.connect(self.VIdeo_button_clicked) #비디오 버튼이벤트 생성

        self.Tf_button = QtWidgets.QPushButton(Dialog)
        self.Tf_button.setGeometry(QtCore.QRect(180, 10, 71, 61))
        self.Tf_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Tf_button.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/tensorflow.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Tf_button.setIcon(icon2)
        self.Tf_button.setIconSize(QtCore.QSize(50, 50))
        self.Tf_button.setObjectName("Tf_button")
        self.Tf_button.clicked.connect(self.Tf_button_clicked) #텐서플로 버튼이벤트 생성
        
        self.Res_button = QtWidgets.QPushButton(Dialog)
        self.Res_button.setGeometry(QtCore.QRect(260, 10, 71, 61))
        self.Res_button.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.Res_button.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap("../My_detection/mygui/image/result.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Res_button.setIcon(icon3)
        self.Res_button.setIconSize(QtCore.QSize(50, 50))
        self.Res_button.setObjectName("Res_button")
        self.Reass_button.clicked.connect(self.Res_button_clicked) #결과창 버튼이벤트 생성
        
        '''라벨생성자'''
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(40, 80, 41, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(120, 80, 41, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(190, 80, 51, 16))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(280, 80, 41, 16))
        self.label_5.setObjectName("label_5")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "오일쇼크"))
        self.label.setText(_translate("Dialog", "카메라"))
        self.label_2.setText(_translate("Dialog", "동영상"))
        self.label_3.setText(_translate("Dialog", "텐서플로"))
        self.label_5.setText(_translate("Dialog", "결과창"))
##end UI set
        
    def Cam_button_clicked(self):

        
        '''
        label_4.setPixmap
        
        _, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        image = qimage2ndarray.array2qimage(frame)  #SOLUTION FOR MEMORY LEAK
        self.label_4.setPixmap(QPixmap.fromImage(image))
        '''
        
        print('씨불탱1')
    #end def Cam_button
        
    def VIdeo_button_clicked(self):
        
        print('씨불탱2')
    #end def Video_button
        
    def Tf_button_clicked(self):
        
        print('씨불탱3')
    #end def Tf_button
        
    def Res_button_clicked(self):
        
        print('씨불탱4')
    #end def Res_button

        
'''
class Detection(self):
'''

    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)


    
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())
