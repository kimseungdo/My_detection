# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mygui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import os
import cv2
import time
from time import sleep
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *
import tensorflow as tf
import pymysql
from threading import Thread
from datetime import datetime
from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils2 as vis_util

'''
객체인식 원형코드 사용시from object_detection.utils import visualization_utils as vis_util
객체인식 변형코드 사용시 from object_detection.utils import visualization_utils2 as vis_util <<<----- 번호판 인식모델
'''
from object_detection.utils.visualization_utils2 import car_info  # <<< utils2 를 설정했을때 사용할것
# from object_detection.utils import ops as utils_ops

conn = pymysql.connect(host='localhost', user='root', password='1234', db='OilShock', charset='utf8')
# host = DB주소(localhost 또는 ip주소), user = DB id, password = DB password, db = DB명

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
now = datetime.now()    # 현재시각 가져오는 메소드

class Ui_Dialog(QWidget, object):

    def setupUi(self, Dialog):
        Dialog.resize(1280, 720)
        Dialog.setStatusTip("")
        Dialog.setStyleSheet("background-color: rgb(255, 255, 255);")
        Dialog.setWindowIcon(QtGui.QIcon('image/123.jpg'))  # WindowIcon 설정
        Dialog.setWindowTitle('Oil Shock - The Fuel Classifier System')
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)

        # Fixed Ui & 배경 라벨
        self.Main_lb = QtWidgets.QLabel(Dialog)
        self.Main_lb.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        pixmap = QPixmap('image/theme.jpg')
        pixmap = pixmap.scaled(1280, 720)  # 사이즈 재설정
        self.Main_lb.setPixmap(pixmap)

        # Intro Ui & 인트로 프레임
        self.Intro_fr = QtWidgets.QFrame(Dialog)
        self.Intro_fr.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        self.Intro_fr.setStyleSheet("background-color: rgb(204, 204, 204, 50);")
        self.Intro_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Intro_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        # 봉유소에 오신 것을 환영합니다. 라벨
        self.Hello_lb = QtWidgets.QLabel(self.Intro_fr)
        self.Hello_lb.setGeometry(QtCore.QRect(263, 245, 754, 54))
        self.Hello_lb.setStyleSheet('background-color: rgb(); font-size: 40pt; font-family: 맑은 고딕;')
        self.Hello_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Hello_lb.setText('봉유소에 오신 것을 환영합니다.')
        # 주유를 하시려면 text 라벨
        self.Intro_lb = QtWidgets.QLabel(self.Intro_fr)
        self.Intro_lb.setGeometry(QtCore.QRect(468, 314, 344, 52))
        self.Intro_lb.setStyleSheet('background-color: rgb(); font-size: 36pt; font-family: 맑은 고딕;')
        self.Intro_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Intro_lb.setText('주유를 하시려면')
        # 아래의 (시 작)버튼을 눌러주세요. text 라벨
        self.Intro_lb1 = QtWidgets.QLabel(self.Intro_fr)
        self.Intro_lb1.setGeometry(QtCore.QRect(295, 374, 690, 52))
        self.Intro_lb1.setStyleSheet('background-color: rgb(); font-size: 36pt; font-family: 맑은 고딕;')
        self.Intro_lb1.setAlignment(QtCore.Qt.AlignCenter)
        self.Intro_lb1.setText('아래의        버튼을 눌러주세요.')
        # (시작)Text 라벨
        self.Start_lb = QtWidgets.QLabel(self.Intro_fr)
        self.Start_lb.setGeometry(QtCore.QRect(450, 384, 120, 42))
        self.Start_lb.setStyleSheet(
            'border : 2px solid black; border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 18pt; font-family: 맑은 고딕;')
        self.Start_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Start_lb.setText('시 작')
        # 시작 버튼
        self.Start_button = QtWidgets.QPushButton(self.Intro_fr)
        self.Start_button.setGeometry(QtCore.QRect(540, 445, 200, 70))
        self.Start_button.setStyleSheet('border : 3px solid black; border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 32pt; font-family: 맑은 고딕;')
        self.Start_button.setText('시 작')
        self.Start_button.clicked.connect(self.Start_button_clicked)  # 버튼이벤트
        # 인트로 프레임 컽
        #self.Intro_fr.setVisible(False)

        # Main Ui & 메인 프레임
        self.Main_fr = QtWidgets.QFrame(Dialog)
        self.Main_fr.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        self.Main_fr.setStyleSheet("background-color: rgb();")
        self.Main_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Main_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Main_fr.setVisible(False)
        # 영상이 나올 라벨
        self.Video_lb = QtWidgets.QLabel(self.Main_fr)
        self.Video_lb.setGeometry(QtCore.QRect(50, 190, 580, 480))
        self.Video_lb.setStyleSheet('border : 4px solid black; border-radius: 10px; background-color: rgb(204, 204, 204, 100); font-size: 30pt; font-family: 맑은 고딕;')  # 폰트&사이즈
        self.Video_lb.setText('여기에 카메라 \n영상이 재생됩니다.')
        self.Video_lb.setAlignment(QtCore.Qt.AlignCenter)  # 중앙 정렬
        # 프레임 라벨
        self.Fps_lb = QtWidgets.QLabel(self.Main_fr)
        self.Fps_lb.setGeometry(QtCore.QRect(55, 643, 81, 16))
        self.Fps_lb.setStyleSheet('background-color: rgb(); color : white; font-size: 10pt; font-family: 맑은 고딕;')
        # 결과창 프레임
        self.frame = QtWidgets.QFrame(self.Main_fr)
        self.frame.setGeometry(QtCore.QRect(650, 190, 580, 480))
        self.frame.setStyleSheet("border-radius: 10px; background-color: rgb(204, 204, 204, 100);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        # 번호판 인식 중.. Text 라벨
        self.Loading_lb = QtWidgets.QLabel(self.frame)
        self.Loading_lb.setGeometry(QtCore.QRect(147, 219, 286, 42))
        self.Loading_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Loading_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Loading_lb.setText('번호판 인식 중..')

        # 유종 정보 등록된 프레임
        self.Ex_fr = QtWidgets.QFrame(Dialog)
        self.Ex_fr.setGeometry(QtCore.QRect(650, 190, 580, 480))
        self.Ex_fr.setStyleSheet("border-radius: 10px; background-color: rgb();")
        self.Ex_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Ex_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        # (고객님의 유종은)Text 라벨
        self.Cus_oil_lb = QtWidgets.QLabel(self.Ex_fr)
        self.Cus_oil_lb.setGeometry(QtCore.QRect(90, 190, 400, 60))
        self.Cus_oil_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Cus_oil_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Cus_oil_lb.setText('고객님 차량의 유종은')
        # 유종 정보 라벨
        self.Oil_type_lb = QtWidgets.QLabel(self.Ex_fr)
        self.Oil_type_lb.setGeometry(QtCore.QRect(97, 240, 264, 60))
        self.Oil_type_lb.setStyleSheet('color : rgb(000, 000, 000); background-color: rgb(); font-weight : bold; font-size: 30pt; font-family: 맑은 고딕;')
        self.Oil_type_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Oil_type_lb.setText('휘발유(가솔린)')
        # (입니다)Text 라벨
        self.Ex_last_lb = QtWidgets.QLabel(self.Ex_fr)
        self.Ex_last_lb.setGeometry(QtCore.QRect(365, 240, 130, 60))
        self.Ex_last_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Ex_last_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Ex_last_lb.setText('입니다.')
        # (유종 정보가 맞다면 (확인)을 눌러주세요.)Text 라벨
        self.Plz_continue_lb = QtWidgets.QLabel(self.Ex_fr)
        self.Plz_continue_lb.setGeometry(QtCore.QRect(73, 320, 434, 30))
        self.Plz_continue_lb.setStyleSheet('background-color: rgb(); font-size: 18pt; font-family: 맑은 고딕;')
        self.Plz_continue_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Plz_continue_lb.setText('유종 정보가 맞다면         을 눌러주세요.')
        # (확인)Text 라벨
        self.Check_lb = QtWidgets.QLabel(self.Ex_fr)
        self.Check_lb.setGeometry(QtCore.QRect(286, 324, 63, 22.5))
        self.Check_lb.setStyleSheet(
            'border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 12pt; font-family: 맑은 고딕;')
        self.Check_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Check_lb.setText('확 인')
        # 확인 버튼
        self.Confirm_button = QtWidgets.QPushButton(self.Ex_fr)
        self.Confirm_button.setGeometry(QtCore.QRect(300, 370, 140, 50))
        self.Confirm_button.setStyleSheet(
            'border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')
        self.Confirm_button.setText('확 인')
        self.Confirm_button.clicked.connect(self.Confirm_button_clicked)  # 확인 버튼이벤트
        # 취소 버튼
        self.CCancel_button = QtWidgets.QPushButton(self.Ex_fr)
        self.CCancel_button.setGeometry(QtCore.QRect(140, 370, 140, 50))
        self.CCancel_button.setStyleSheet('border-radius: 5px; background-color: rgb(051, 051, 051); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')
        self.CCancel_button.setText('취 소')
        self.CCancel_button.clicked.connect(self.Cancel_button_clicked)  # 취소 버튼이벤트
        self.Ex_fr.setVisible(False)
        # 유종 정보 등록 프레임 컽

        # 유종 정보 미등록 프레임
        self.Regi_fr = QtWidgets.QFrame(Dialog)
        self.Regi_fr.setGeometry(QtCore.QRect(650, 190, 580, 480))
        self.Regi_fr.setStyleSheet("border-radius: 10px; background-color: rgb();")
        self.Regi_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Regi_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        # (고객님의 차량은)Text 라벨
        self.Cus_oil_none_lb = QtWidgets.QLabel(self.Regi_fr)
        self.Cus_oil_none_lb.setGeometry(QtCore.QRect(145, 190, 290, 60))
        self.Cus_oil_none_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Cus_oil_none_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Cus_oil_none_lb.setText('고객님의 차량은')
        # (유종 정보가 없습니다)Text 라벨
        self.Oil_type_none_lb = QtWidgets.QLabel(self.Regi_fr)
        self.Oil_type_none_lb.setGeometry(QtCore.QRect(94, 240, 392, 60))
        self.Oil_type_none_lb.setStyleSheet('color : rgb(000, 000, 000); background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Oil_type_none_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Oil_type_none_lb.setText('유종 정보가 없습니다.')
        # (유종 정보를 등록하시려면 (등록하기)를 눌러주세요.)Text 라벨
        self.Plz_register_lb = QtWidgets.QLabel(self.Regi_fr)
        self.Plz_register_lb.setGeometry(QtCore.QRect(33, 320, 514, 30))
        self.Plz_register_lb.setStyleSheet('background-color: rgb(); font-size: 18pt; font-family: 맑은 고딕;')
        self.Plz_register_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Plz_register_lb.setText('유종 정보를 등록하시려면          를 눌러주세요.')
        # (등록하기)Text 라벨
        self.Register_lb = QtWidgets.QLabel(self.Regi_fr)
        self.Register_lb.setGeometry(QtCore.QRect(319, 324, 70, 25))
        self.Register_lb.setStyleSheet('border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 12pt; font-family: 맑은 고딕;')
        self.Register_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Register_lb.setText('등록하기')
        # 등록 버튼
        self.Register_button = QtWidgets.QPushButton(self.Regi_fr)
        self.Register_button.setGeometry(QtCore.QRect(300, 370, 140, 50))
        self.Register_button.setStyleSheet('border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 20pt; font-family: 맑은 고딕;')
        self.Register_button.setText('등록하기')
        self.Register_button.clicked.connect(self.Register_button_clicked)  # 등록 버튼이벤트
        # 취소 버튼
        self.RCancel_button = QtWidgets.QPushButton(self.Regi_fr)
        self.RCancel_button.setGeometry(QtCore.QRect(140, 370, 140, 50))
        self.RCancel_button.setStyleSheet('border-radius: 5px; background-color: rgb(051, 051, 051); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')  # 255 102 051
        self.RCancel_button.setText('취 소')
        self.RCancel_button.clicked.connect(self.Cancel_button_clicked)  # 취소 버튼이벤트
        self.Regi_fr.setVisible(False)
        # 유종 정보 미등록 프레임 컽

        # 이미지, 번호판 프레임
        self.Rema_fr = QtWidgets.QFrame(self.Main_fr)
        self.Rema_fr.setGeometry(QtCore.QRect(650, 190, 580, 480))
        self.Rema_fr.setStyleSheet("border-radius: 10px; background-color: rgb(204, 204, 204, 100);")
        self.Rema_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Rema_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Rema_fr.setVisible(False)
        # 번호판 이미지 라벨
        self.Plate_img_lb = QtWidgets.QLabel(self.Rema_fr)
        self.Plate_img_lb.setGeometry(QtCore.QRect(160, 60, 260, 50))
        self.Plate_img_lb.setStyleSheet('background-color: rgb(000, 000, 000);')
        # pixmap = QPixmap('00.jpg')
        # pixmap = pixmap.scaled(140, 140)
        # self.Plate_img_lb.setPixmap(pixmap)
        # 번호판 라벨
        self.Num_Plate_lb = QtWidgets.QLabel(self.Rema_fr)
        self.Num_Plate_lb.setGeometry(QtCore.QRect(135, 125, 310, 60))  # 785 290 31 60
        self.Num_Plate_lb.setStyleSheet('background-color: rgb(); font-weight : bold; font-size: 36pt; font-family: 맑은 고딕;')
        self.Num_Plate_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Num_Plate_lb.setText('')
        # 메인 프레임 컽

        # Ending Ui & 마지막 프레임
        self.End_fr = QtWidgets.QFrame(Dialog)
        self.End_fr.setGeometry(QtCore.QRect(0, 0, 1280, 720))
        self.End_fr.setStyleSheet("background-color: rgb(204, 204, 204, 50);")
        self.End_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.End_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        # 주유가 완료되었습니다. text 라벨
        self.End_lb = QtWidgets.QLabel(self.End_fr)
        self.End_lb.setGeometry(QtCore.QRect(362, 299, 556, 54))
        self.End_lb.setStyleSheet('background-color: rgb(); font-size: 40pt; font-family: 맑은 고딕;')
        self.End_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.End_lb.setText('주유가 완료되었습니다.')
        # 안전운행하세요. text 라벨
        self.End1_lb = QtWidgets.QLabel(self.End_fr)
        self.End1_lb.setGeometry(QtCore.QRect(448, 367, 384, 54))
        self.End1_lb.setStyleSheet('background-color: rgb(); font-size: 40pt; font-family: 맑은 고딕;')
        self.End1_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.End1_lb.setText('안전운행하세요.')
        self.End_fr.setVisible(False)
        # 엔드 프레임 컽

        # 유종 등록 프레임
        self.Register_fr = QtWidgets.QFrame(Dialog)
        self.Register_fr.setGeometry(QtCore.QRect(650, 190, 580, 480))
        self.Register_fr.setStyleSheet("border-radius: 10px; background-color: rgb(204, 204, 204, 100);")
        self.Register_fr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Register_fr.setFrameShadow(QtWidgets.QFrame.Raised)
        # 휘발유 버튼
        self.Gasoline_btn = QtWidgets.QPushButton(self.Register_fr)
        self.Gasoline_btn.setGeometry(QtCore.QRect(100, 70, 180, 180))
        self.Gasoline_btn.setStyleSheet('border-radius: 5px; background-color: rgb(255, 255, 000, 200); color : rgb(0, 0, 0); font-weight : bold; font-size: 24pt; font-family: 맑은 고딕;')
        self.Gasoline_btn.setText('휘발유\n(가솔린)')
        self.Gasoline_btn.clicked.connect(self.Gasoline_btn_clicked)  # 가솔린 setText 버튼이벤트
        # 경유 버튼
        self.Diesel_btn = QtWidgets.QPushButton(self.Register_fr)
        self.Diesel_btn.setGeometry(QtCore.QRect(300, 70, 180, 180))
        self.Diesel_btn.setStyleSheet('border-radius: 5px; background-color: rgb(051, 204, 051, 200); color : rgb(0, 0, 0); font-weight : bold; font-size: 24pt; font-family: 맑은 고딕;')  # 102 204 102
        self.Diesel_btn.setText('경 유\n(디 젤)')
        self.Diesel_btn.clicked.connect(self.Diesel_btn_clicked)  # 디젤 setText 버튼이벤트
        # 차량번호 라벨
        self.Regi_NumPlate_lb = QtWidgets.QLabel(self.Register_fr)
        self.Regi_NumPlate_lb.setGeometry(QtCore.QRect(143, 263, 294, 40))
        self.Regi_NumPlate_lb.setStyleSheet('background-color: rgb(); font-weight : bold; font-size: 22pt; font-family: 맑은 고딕;')
        self.Regi_NumPlate_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Regi_NumPlate_lb.setText('차량번호 : 123가4567')
        # 선택 유종 라벨
        self.Choice_Oil_Type_lb = QtWidgets.QLabel(self.Register_fr)
        self.Choice_Oil_Type_lb.setGeometry(QtCore.QRect(156, 301, 278, 40))
        self.Choice_Oil_Type_lb.setStyleSheet('background-color: rgb(); font-weight : bold; font-size: 22pt; font-family: 맑은 고딕;')
        self.Choice_Oil_Type_lb.setAlignment(QtCore.Qt.AlignCenter)
        # 등록 버튼
        self.Regi_DB_button = QtWidgets.QPushButton(self.Register_fr)
        self.Regi_DB_button.setGeometry(QtCore.QRect(300, 360, 140, 50))
        self.Regi_DB_button.setStyleSheet('border-radius: 5px; background-color: rgb(000, 153, 153); color : rgb(255, 255, 255); font-size: 20pt; font-family: 맑은 고딕;')
        self.Regi_DB_button.setText('등록')
        self.Regi_DB_button.clicked.connect(self.Regi_DB_clicked)  # 등록 버튼이벤트
        # 취소 버튼
        self.ReCancel_button = QtWidgets.QPushButton(self.Register_fr)
        self.ReCancel_button.setGeometry(QtCore.QRect(140, 360, 140, 50))
        self.ReCancel_button.setStyleSheet('border-radius: 5px; background-color: rgb(051, 051, 051); color : rgb(255, 255, 255); font-size: 24pt; font-family: 맑은 고딕;')  # 255 102 051
        self.ReCancel_button.setText('취소')
        self.ReCancel_button.clicked.connect(self.Cancel_button_clicked)  # 취소 버튼이벤트
        # 성공적으로 text 라벨
        self.Saving_lb = QtWidgets.QLabel(self.Register_fr)
        self.Saving_lb.setGeometry(QtCore.QRect(190, 197, 200, 42))
        self.Saving_lb.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Saving_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Saving_lb.setText('성공적으로')
        # 등록되었습니다. 라벨
        self.Saving_lb1 = QtWidgets.QLabel(self.Register_fr)
        self.Saving_lb1.setGeometry(QtCore.QRect(146, 241, 288, 42))
        self.Saving_lb1.setStyleSheet('background-color: rgb(); font-size: 30pt; font-family: 맑은 고딕;')
        self.Saving_lb1.setAlignment(QtCore.Qt.AlignCenter)
        self.Saving_lb1.setText('등록되었습니다.')
        self.Saving_lb.setVisible(False)
        self.Saving_lb1.setVisible(False)
        self.Register_fr.setVisible(False)
        # 유종 등록 프레임 끝

        # 로고 이미지 라벨
        self.Logo_lb = QtWidgets.QLabel(Dialog)
        self.Logo_lb.setGeometry(QtCore.QRect(38, 40, 279, 101))
        self.Logo_lb.setStyleSheet("background-color: rgb()")
        pixmap = QPixmap('image/logo.png')
        #pixmap = pixmap.scaled(279, 101)  # 사이즈 재설정
        self.Logo_lb.setPixmap(pixmap)
        # 제작자 라벨
        self.Maker_lb = QtWidgets.QLabel(Dialog)
        self.Maker_lb.setGeometry(QtCore.QRect(990, 694, 274, 16))
        self.Maker_lb.setStyleSheet('background-color: rgb(); font-size: 10pt; font-family: 맑은 고딕;')
        self.Maker_lb.setAlignment(QtCore.Qt.AlignRight)
        self.Maker_lb.setText('The Fuel Classifier System  |  Team. Oil Shock')

    def setImage(self, image):  # 이미지를 라벨에 넣는 함수
        ui.Video_lb.setPixmap(QtGui.QPixmap.fromImage(image))

    # Event 함수
    def Start_button_clicked(self):  # 시작 버튼 이벤트
        self.Intro_fr.setVisible(False) # 인트로 프레임 Visible = False
        self.Main_fr.setVisible(True)   # 메인 프레임 Visible = True
        self.frame.setVisible(True)
        global wtf  # 반복문을 제어할 변수
        wtf = 1
        th1 = Thread(self)
        th1.changePixmap.connect(self.setImage)
        th2 = Thread2(self)

        th1.start() # 스레드1 스타트, 동영상을 라벨에 올리는 스레드
        th2.start() # 스레드2 스타트, 객체 인식 및 번호판 인식 스레드

    def Regi_SetUi(self, bool): # Register Ui 설정 함수
        self.Gasoline_btn.setVisible(bool)
        self.Diesel_btn.setVisible(bool)
        self.Regi_NumPlate_lb.setVisible(bool)
        self.Choice_Oil_Type_lb.setVisible(bool)
        self.Regi_DB_button.setVisible(bool)
        self.ReCancel_button.setVisible(bool)
        self.Register_fr.setVisible(bool)

    def Register_button_clicked(self):  # 등록 버튼 이벤트(화면 전환)
        global wtf
        wtf = 0
        self.Regi_fr.setVisible(False)
        self.Saving_lb.setVisible(False)
        self.Saving_lb1.setVisible(False)
        self.Regi_SetUi(True)

    def Regi_End_SetUi(self):
        self.Regi_SetUi(False)
        self.Saving_lb.setVisible(True)
        self.Saving_lb1.setVisible(True)
        k = cv2.waitKey(2000)
        self.Main_fr.setVisible(False)
        self.Register_fr.setVisible(False)
        self.End_fr.setVisible(True)
        k = cv2.waitKey(4000)  # 4초 대기
        self.End_fr.setVisible(False)
        self.Intro_fr.setVisible(True)
        global wtf
        wtf = 0

    def Regi_DB_clicked(self):  # 등록 버튼 이벤트(DB에 추가)
        oilType = ''    # 유종 변수
        if self.Choice_Oil_Type_lb.text() == '유종 : 휘발유(가솔린)':   # 휘발유 선택시 이벤트 처리
            oilType = 'G'
            regi_sql = 'INSERT INTO Register VALUES (' + '0' + ', ' + "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second) + ", ""'" + self.Num_Plate_lb.text() + "', ""'" + oilType + "'"", '" + 'path' + "'"')'
            curs = conn.cursor()
            curs.execute(regi_sql)
            conn.commit()
            self.Regi_End_SetUi()
        elif self.Choice_Oil_Type_lb.text() == '유종 : 경유(디젤)':   # 경유 선택시 이벤트 처리
            oilType = 'D'
            regi_sql = 'INSERT INTO Register VALUES (' + '0' + ', ' + "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second) + ", ""'" + self.Num_Plate_lb.text() + "', ""'" + oilType + "'"", '" + 'path' + "'"')'
            curs = conn.cursor()
            curs.execute(regi_sql)
            conn.commit()
            self.Regi_End_SetUi()
        else:   # 유종 미선택 에러 메시지박스
            msg = QMessageBox()
            msg.setWindowTitle('오류')
            msg.setText('유종을 선택해 주세요.')
            msg.setStandardButtons(QMessageBox.Ok)
            result = msg.exec_()
        self.Choice_Oil_Type_lb.setText('') # 선택 유종 초기화

    def Cancel_button_clicked(self):  # 취소 버튼 이벤트
        global wtf
        wtf = 0
        self.Ex_fr.setVisible(False)
        self.Regi_fr.setVisible(False)
        self.Register_fr.setVisible(False)
        self.Rema_fr.setVisible(False)
        self.frame.setVisible(True)
        self.Main_fr.setVisible(False)
        self.Intro_fr.setVisible(True)

    def Confirm_button_clicked(self):  # 확인 버튼 이벤트
        global wtf
        wtf = 0
        oiltype = ''
        if self.Oil_type_lb.text() == '휘발유(가솔린)':
            oiltype = 'G'
        elif self.Oil_type_lb.text() == '경유(디젤)':
            oiltype = 'D'
        refuel_sql = 'INSERT INTO refuelLog VALUES (' + '0' + ', ' + "%04d%02d%02d%02d%02d%02d" % (now.year, now.month, now.day, now.hour, now.minute, now.second) + ", ""'" + self.Num_Plate_lb.text() + "', ""'" + oiltype + "'"", " + 'null' + ", " + 'null' ')' # path는 이미지 경로 추가 요함
        curs = conn.cursor()
        curs.execute(refuel_sql)
        conn.commit()
        self.Ex_fr.setVisible(False)
        self.Main_fr.setVisible(False)
        self.Rema_fr.setVisible(False)
        self.End_fr.setVisible(True)
        k = cv2.waitKey(4000)  # 4초 대기
        self.End_fr.setVisible(False)
        self.Intro_fr.setVisible(True)

    def Gasoline_btn_clicked(self): # 유종 선택 버튼(휘발유)
        self.Choice_Oil_Type_lb.setText('유종 : 휘발유(가솔린)')

    def Diesel_btn_clicked(self): # 유종 선택 버튼(경유)
        self.Choice_Oil_Type_lb.setText('유종 : 경유(디젤)')

class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        prevtime = 0

        while wtf:
            k = cv2.waitKey(30) # 영상 프레임 속도 조절
            ret, frame = capture.read()
            global re, fr
            re = ret
            fr = frame

            # 프레임 표시
            curtime = time.time()
            sec = curtime - prevtime
            prevtime = curtime
            fps = 1 / sec
            str = "FPS : %0.1f" % fps
            ui.Fps_lb.setText(str) # 프레임 라벨에 지속적으로 갱신

            if ret: # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            sleep(0)


class Thread2(QThread):

    def run(self):
        time1 = time.time()
        MIN_ratio = 0.60

        MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        #MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
        GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
        LABEL_FILE = 'data/mscoco_label_map.pbtxt'
        NUM_CLASSES = 90
        # end define

        label_map = lmu.load_labelmap(LABEL_FILE)
        categories = lmu.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        categories_index = lmu.create_category_index(categories)

        print("call label_map & categories : %0.5f" % (time.time() - time1))

        graph_file = MODEL_NAME + '/' + GRAPH_FILE_NAME

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sses = tf.Session(graph=detection_graph)

        print("store in memoey time : %0.5f" % (time.time() - time1))

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        print("road Video time : %0.5f" % (time.time() - time1))

        while wtf:
            global re, fr
            ret = re
            frame = fr
            frame_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, nums) = sses.run(  # np.ndarray
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded}
            )  # end sses.run()

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                categories_index,
                use_normalized_coordinates=True,
                min_score_thresh=MIN_ratio,  # 최소 인식률
                line_thickness=2)  # 선두께

            try:
                curs = conn.cursor()
                sql = 'SELECT oilType from Register Where carNum = ' + "'" + str(car_info[0]) + "'"  # 실행 할 쿼리문 입력
                print(sql)
                curs.execute(sql)  # 쿼리문 실행
                rows = curs.fetchone()  # 데이터 패치

                if rows == ('G',):
                    ui.Oil_type_lb.setText('휘발유(가솔린)')
                    ui.Oil_type_lb.setGeometry(QtCore.QRect(97, 240, 264, 60))
                    ui.Ex_last_lb.setGeometry(QtCore.QRect(365, 240, 130, 60))
                    self.Set_Img_Numlb()
                    self.Ex_Show_Frame()
                elif rows == ('D',):
                    ui.Oil_type_lb.setText('경유(디젤)')
                    ui.Oil_type_lb.setGeometry(QtCore.QRect(131, 240, 184, 60))
                    ui.Ex_last_lb.setGeometry(QtCore.QRect(319, 240, 130, 60))
                    self.Set_Img_Numlb()
                    self.Ex_Show_Frame()
                else:
                    self.Set_Img_Numlb()
                    ui.Regi_NumPlate_lb.setText(str(car_info[0]))
                    ui.frame.setVisible(False)
                    ui.Ex_fr.setVisible(False)
                    ui.Rema_fr.setVisible(True)
                    ui.Regi_fr.setVisible(True)
                    if wtf == 0:
                        ui.Rema_fr.setVisible(False)
                        ui.Regi_fr.setVisible(False)
            except:
                pass
            sleep(0)

    def Set_Img_Numlb(self): # 번호판 이미지와 인식 문자열 라벨에 올리는 함수
        pixmap = QPixmap('00.jpg')
        pixmap = pixmap.scaled(260, 50)
        ui.Num_Plate_lb.setText(str(car_info[0]))
        ui.Plate_img_lb.setPixmap(pixmap)

    def Ex_Show_Frame(self):
        ui.frame.setVisible(False)
        ui.Regi_fr.setVisible(False)
        ui.Rema_fr.setVisible(True)
        ui.Ex_fr.setVisible(True)
        if wtf == 0:
            ui.Rema_fr.setVisible(False)
            ui.Ex_fr.setVisible(False)


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    changePixmap = pyqtSignal(QImage)

    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()

    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("asdf.mp4")  # 165145 162900

    sys.exit(app.exec_())
