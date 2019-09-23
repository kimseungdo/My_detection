# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mygui.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!
import os
import cv2
import time
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread
from multiprocessing import Process, Queue
from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

from time import sleep

# file import
import NumberPlate as NP

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *


class Ui_Dialog(QWidget, object):

    def setupUi(self, Dialog):
        Dialog.resize(1280, 720)
        Dialog.setStatusTip("")
        Dialog.setStyleSheet("background-color: rgb(255, 255, 255);")
        Dialog.setWindowIcon(QtGui.QIcon('../NumPlate/image/123.jpg')) # WindowIcon 설정
        Dialog.setWindowTitle('Oil Shock - The Oil Kind Determination System')
        Dialog.setSizeGripEnabled(False)
        Dialog.setModal(False)

        self.frame = QtWidgets.QFrame(Dialog) # 결과창 프레임
        self.frame.setGeometry(QtCore.QRect(550, 190, 700, 500))
        self.frame.setStyleSheet("background-color: rgb(255, 255, 255);")  # 204 255 255 // 102 204 255
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)

        self.textBrowser = QtWidgets.QTextBrowser(self.frame) # 결과창 border용으로 냅둠
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 700, 500))
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Raised)

        '''버튼&아이콘 생성자'''
        # 시작 버튼
        self.Rec_button = QtWidgets.QPushButton(Dialog)
        self.Rec_button.setGeometry(QtCore.QRect(1150, 30, 100, 100))
        self.Rec_button.setStyleSheet('background-color: rgb(240, 240, 240);')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('../Numplate/image/play-button.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.Rec_button.setIcon(icon)
        self.Rec_button.setIconSize(QtCore.QSize(50, 50))
        self.Rec_button.clicked.connect(self.Rec_button_clicked)  # 카메라 버튼이벤트 생성

        # 등록 버튼
        self.Register_button = QtWidgets.QPushButton(Dialog)
        self.Register_button.setGeometry(QtCore.QRect(650, 575, 140, 75))
        self.Register_button.setStyleSheet(
            'background-color: rgb(000, 102, 255); color : rgb(255, 255, 255); font-size: 24pt; font-family: 메이플스토리;')  # 255 102 051
        self.Register_button.setText('등록하기')
        self.Register_button.clicked.connect(self.Register_button_clicked)  # 취소 버튼이벤트 생성

        # 취소 버튼
        self.Cancel_button = QtWidgets.QPushButton(Dialog)
        self.Cancel_button.setGeometry(QtCore.QRect(830, 575, 140, 75))
        self.Cancel_button.setStyleSheet('background-color: rgb(255, 051, 051); color : rgb(255, 255, 255); font-size: 24pt; font-family: 메이플스토리;') # 255 102 051
        self.Cancel_button.setText('취 소')
        self.Cancel_button.clicked.connect(self.Cancel_button_clicked)  # 취소 버튼이벤트 생성

        # 확인 버튼
        self.Confirm_button = QtWidgets.QPushButton(Dialog)
        self.Confirm_button.setGeometry(QtCore.QRect(1010, 575, 140, 75))
        self.Confirm_button.setStyleSheet('background-color: rgb(10, 204, 102); color : rgb(255, 255, 255); font-size: 24pt; font-family: 메이플스토리;')
        self.Confirm_button.setText('확 인')
        self.Confirm_button.clicked.connect(self.Confirm_button_clicked) # 확인 버튼이벤트 생성

        '''라벨생성자'''
        # 로고 이미지 라벨
        self.Logo_lb = QtWidgets.QLabel(Dialog)
        self.Logo_lb.setGeometry(QtCore.QRect(30, 30, 140, 140))
        pixmap = QPixmap('../NumPlate/image/logo.png')
        #pixmap = pixmap.scaled(140, 140) # 사이즈 재설정
        self.Logo_lb.setPixmap(pixmap)

        # 번호판 이미지 라벨
        self.Plate_img_lb = QtWidgets.QLabel(Dialog)
        self.Plate_img_lb.setGeometry(QtCore.QRect(900, 50, 200, 50))
        self.Plate_img_lb.setStyleSheet('background-color: rgb(000, 000, 000);')
        #pixmap = QPixmap('00.jpg')
        #pixmap = pixmap.scaled(140, 140)
        #self.Plate_img_lb.setPixmap(pixmap)

        # 번호판 이미지 설명 라벨
        self.Plate_terri_lb = QtWidgets.QLabel(Dialog)
        self.Plate_terri_lb.setGeometry(QtCore.QRect(900, 100, 200, 50))  # 225 40 250 50
        self.Plate_terri_lb.setStyleSheet('font-size: 14pt; font-family: 메이플스토리;')
        self.Plate_terri_lb.setText('영상처리 된 번호판 이미지')
        self.Plate_terri_lb.setAlignment(QtCore.Qt.AlignCenter)

        # 디자인용 선 라벨
        self.Line_lb = QtWidgets.QLabel(Dialog)
        self.Line_lb.setGeometry(QtCore.QRect(170, 160, 1080, 4))
        self.Line_lb.setStyleSheet('background-color: rgb(000, 000, 000);')

        # 영상이 나올 라벨
        self.Video_lb = QtWidgets.QLabel(Dialog)
        self.Video_lb.setGeometry(QtCore.QRect(30, 190, 480, 500))
        self.Video_lb.setStyleSheet(
            'background-color: rgb(153, 153, 153); font-size: 24pt; font-family: 메이플스토리;')  # 폰트&사이즈
        self.Video_lb.setText('여기에 카메라 \n영상이 재생됩니다.')
        self.Video_lb.setAlignment(QtCore.Qt.AlignCenter)  # 중앙 정렬

        # 번호판 라벨
        self.Num_Plate_lb = QtWidgets.QLabel(Dialog)
        self.Num_Plate_lb = QtWidgets.QLabel(self.frame)
        self.Num_Plate_lb.setGeometry(QtCore.QRect(195, 40, 310, 60)) # 225 40 250 50
        self.Num_Plate_lb.setStyleSheet('border: 2px solid black; font-size: 42pt; font-family: 메이플스토리;')
        self.Num_Plate_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Num_Plate_lb.setText('인식한 번호판')

        # (고객님의 유종은)Text 라벨
        self.Cus_oil_lb = QtWidgets.QLabel(Dialog)
        self.Cus_oil_lb = QtWidgets.QLabel(self.frame)
        self.Cus_oil_lb.setGeometry(QtCore.QRect(100, 120, 500, 60))
        self.Cus_oil_lb.setStyleSheet('font-size: 42pt; font-family: 메이플스토리;')
        self.Cus_oil_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Cus_oil_lb.setText('고객님 차량의 유종은')

        # 유종 정보 라벨
        self.Oil_kind_lb = QtWidgets.QLabel(Dialog)
        self.Oil_kind_lb = QtWidgets.QLabel(self.frame)
        self.Oil_kind_lb.setGeometry(QtCore.QRect(95, 200, 350, 60))
        self.Oil_kind_lb.setStyleSheet('color : rgb(255, 000, 000); font-size: 42pt; font-family: 메이플스토리;')
        self.Oil_kind_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Oil_kind_lb.setText('휘발유(가솔린)')

        # (입니다)Text 라벨
        self.Ex_last_lb = QtWidgets.QLabel(Dialog)
        self.Ex_last_lb = QtWidgets.QLabel(self.frame)
        self.Ex_last_lb.setGeometry(QtCore.QRect(445, 200, 160, 60))
        self.Ex_last_lb.setStyleSheet('font-size: 42pt; font-family: 메이플스토리;')
        self.Ex_last_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Ex_last_lb.setText('입니다.')

        # (유종 정보가 맞다면 (확인)을 눌러주세요.)Text 라벨
        self.Plz_continue_lb = QtWidgets.QLabel(Dialog)
        self.Plz_continue_lb = QtWidgets.QLabel(self.frame)
        self.Plz_continue_lb.setGeometry(QtCore.QRect(167, 290, 365, 30))
        self.Plz_continue_lb.setStyleSheet('font-size: 18pt; font-family: 메이플스토리;')
        self.Plz_continue_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Plz_continue_lb.setText('유종 정보가 맞다면          을 눌러주세요.')

        # (확인)Text 라벨
        self.Check_lb = QtWidgets.QLabel(Dialog)
        self.Check_lb = QtWidgets.QLabel(self.frame)
        self.Check_lb.setGeometry(QtCore.QRect(350, 290, 50, 30))
        self.Check_lb.setStyleSheet('background-color: rgb(10, 204, 102); color : rgb(255, 255, 255); font-size: 18pt; font-family: 메이플스토리;')
        self.Check_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Check_lb.setText('확 인')

        # (유종 정보가 없다면 (등록하기)를 눌러주세요.)Text 라벨
        self.Plz_register_lb = QtWidgets.QLabel(Dialog)
        self.Plz_register_lb = QtWidgets.QLabel(self.frame)
        self.Plz_register_lb.setGeometry(QtCore.QRect(150, 330, 400, 30))
        self.Plz_register_lb.setStyleSheet('font-size: 18pt; font-family: 메이플스토리;')
        self.Plz_register_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Plz_register_lb.setText('유종 정보가 없다면               를 눌러주세요.')

        # (등록하기)Text 라벨
        self.Register_lb = QtWidgets.QLabel(Dialog)
        self.Register_lb = QtWidgets.QLabel(self.frame)
        self.Register_lb.setGeometry(QtCore.QRect(335, 330, 80, 30))
        self.Register_lb.setStyleSheet(
            'background-color: rgb(000, 102, 255); color : rgb(255, 255, 255); font-size: 18pt; font-family: 메이플스토리;')
        self.Register_lb.setAlignment(QtCore.Qt.AlignCenter)
        self.Register_lb.setText('등록하기')

        # 프레임 라벨
        self.Fps_lb = QtWidgets.QLabel(Dialog)
        self.Fps_lb.setGeometry(QtCore.QRect(30, 695, 81, 16))
        self.Fps_lb.setStyleSheet('font-size: 10pt; font-family: 메이플스토리;')
        self.Fps_lb.setText('프레임')

        # 제작자 라벨
        self.Maker_lb = QtWidgets.QLabel(Dialog)
        self.Maker_lb.setGeometry(QtCore.QRect(900, 695, 350, 16))
        self.Maker_lb.setStyleSheet('font-size: 10pt; font-family: 메이플스토리;')
        self.Maker_lb.setAlignment(QtCore.Qt.AlignRight)
        self.Maker_lb.setText('The Oil Kind Determination System  ⓟTeam Oil Shock')

    def setImage(self, image): # 이미지를 라벨에 넣는 함수
        ui.Video_lb.setPixmap(QtGui.QPixmap.fromImage(image))

    # Event 함수
    def Rec_button_clicked(self): # 시작 버튼 이벤트
        th1 = Thread(self)
        th1.changePixmap.connect(self.setImage)
        th2 = Thread2(self)

        th1.start()
        th2.start()

        print('영상 재생')

    def Register_button_clicked(self): # 등록 버튼 이벤트
        print('등록')

    def Cancel_button_clicked(self): # 취소 버튼 이벤트
        print('취소')

    def Confirm_button_clicked(self): # 확인 버튼 이벤트
        print('확인')


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def run(self):
        sleep(6)

        prevtime = 0

        while True:
            ret, frame = capture.read()

            # 프레임 표시
            curtime = time.time()
            sec = curtime - prevtime
            prevtime = curtime
            fps = 1 / sec
            str = "FPS : %0.1f" % fps
            cv2.putText(frame, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            ui.Fps_lb.setText(str)
            # end 프레임

            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break



class Thread2(QThread):

    def run(self):
        time1 = time.time()
        MIN_ratio = 0.9

        # MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
        MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
        GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
        LABEL_FILE = 'data/mscoco_label_map.pbtxt'
        NUM_CLASSES = 90
        # end define

        label_map = lmu.load_labelmap(LABEL_FILE)
        categories = lmu.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        categories_index = lmu.create_category_index(categories)

        print("call label_map & categories : %0.5f" % (time.time() - time1))

        graph_file = MODEL_NAME + '/' + GRAPH_FILE_NAME

        # thread function
        def find_detection_target(categories_index, classes, scores):
            time1_1 = time.time()  # 스레드함수 시작시간
            print("스레드 시작")

            objects = []  # 리스트 생성
            for index, value in enumerate(classes[0]):
                object_dict = {}  # 딕셔너리
                if scores[0][index] > MIN_ratio:
                    object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                        scores[0][index]
                    objects.append(object_dict)  # 리스트 추가
            print(objects)

            print("스레드 함수 처리시간 %0.5f" & (time.time() - time1_1))

        # end thread function

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

        print("make tensor time : %0.5f" % (time.time() - time1))

        prevtime = 0

        # thread_1 = Process(target = find_detection_target, args = (categories_index, classes, scores))#쓰레드 생성
        print("road Video time : %0.5f" % (time.time() - time1))

        while True:
            ret, frame = capture.read()
            frame_expanded = np.expand_dims(frame, axis=0)
            height, width, channel = frame.shape

            (boxes, scores, classes, nums) = sses.run(  # np.ndarray
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded}
            )  # end sses.run()

            # objects = [] #리스트 생성
            for index, value in enumerate(classes[0]):
                object_dict = {}  # 딕셔너리
                if scores[0][index] > MIN_ratio:
                    object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                        scores[0][index]
                    # objects.append(object_dict) #리스트 추가

                    # visualize_boxes_and_labels_on_image_array box_size_info 이미지 정
                    # for box, color in box_to_color_map.items():
                    #    ymin, xmin, ymax, xmax = box
                    # [index][0] [1]   [2]  [3]

                    ymin = int((boxes[0][index][0] * height))
                    xmin = int((boxes[0][index][1] * width))
                    ymax = int((boxes[0][index][2] * height))
                    xmax = int((boxes[0][index][3] * width))

                    Result = frame[ymin:ymax, xmin:xmax]
                    cv2.imwrite('car.jpg', Result)
                    try:
                        result_chars = NP.number_recognition('car.jpg')
                        ui.Num_Plate_lb.setText(result_chars)

                        pixmap = QPixmap('00.jpg')
                        pixmap = pixmap.scaled(200, 50)
                        ui.Plate_img_lb.setPixmap(pixmap)
                        # print(NP.check())

                    except:
                        print("응안돼")

            # print(objects)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    changePixmap = pyqtSignal(QImage)

    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()

    # capture = cv2.VideoCapture(0)
    capture = cv2.VideoCapture("20190916_165145.mp4")  # 165145 162900

    sys.exit(app.exec_())
