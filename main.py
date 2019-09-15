import cv2
import os, sys
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

#file import
from mygui import Ui_Dialog

'''
Cam_button
Video_button
Tf_button
Res_button

label_4 >> 영상 결과


self.Cam_button.clicked.connect(self.Cam_button_clicked)
self.VIdeo_button.clicked.connect(self.VIdeo_button_clicked)
self.Tf_button.clicked.connect(self.Tf_button_clicked)
self.Res_button.clicked.connect(self.Res_button_clicked)

'''
class button_trig(
        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)


    
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    
    Dialog.show()
    sys.exit(app.exec_())
