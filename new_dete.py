import os
import cv2
import time
import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

from queue import Queue
from threading import Thread
from multiprocessing import Process


from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils2 as vis_util
from object_detection.utils.visualization_utils2 import car_info
'''
객체인식 원형코드 사용시from object_detection.utils import visualization_utils as vis_util
객체인식 변형코드 사용시 from object_detection.utils import visualization_utils2 as vis_util <<<----- 번호판 인식모델
'''
from object_detection.utils import ops as utils_ops

#file import
import NumberPlate as NP



#define
time1 = time.time()
MIN_ratio = 0.65

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
LABEL_FILE = 'data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90
#end define

label_map = lmu.load_labelmap(LABEL_FILE)
categories = lmu.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
categories_index = lmu.create_category_index(categories)

print("call label_map & categories : %0.5f" % (time.time() - time1))

graph_file = MODEL_NAME + '/' + GRAPH_FILE_NAME
#thread function
def find_detection_target(categories_index, classes, scores):
    time1_1 = time.time() #스레드함수 시작시간
    print("스레드 시작")
    
    objects = [] #리스트 생성
    for index, value in enumerate(classes[0]):
        object_dict = {} #딕셔너리
        if scores[0][index] > MIN_ratio:
            object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                    scores[0][index]
            objects.append(object_dict) #리스트 추가
    print(objects)
    
    print("스레드 함수 처리시간 %0.5f" & (time.time() - time1_1))

#end thread function



detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(graph_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name = '')

    sses = tf.Session(graph = detection_graph)

print("store in memoey time : %0.5f" % (time.time() - time1))

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

print("make tensor time : %0.5f" % (time.time() - time1))

    
#capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('20190916_162900.mp4')
capture = cv2.VideoCapture("변환/사기각(HD_8fps_110s).mp4")
prevtime = 0

#thread_1 = Process(target = find_detection_target, args = (categories_index, classes, scores))#쓰레드 생성
print("road Video time : %0.5f" % (time.time() - time1))


'''
Video_switch = True
while Video_switch:
    ret, frame = capture.read()
    key = cv2.waitKey(1) & 0xFF
    
    height, width, channel = frame.shape
    frame_expanded = np.expand_dims(frame, axis = 0)

    #프레임 표시
    curtime = time.time()
    sec = curtime - prevtime
    prevtime = curtime
    fps = 1/ sec
    str = "FPS : %0.1f" % fps
    cv2.putText(frame, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    #end 프레임
    
    (boxes, scores, classes, nums) = sses.run(#np.ndarray
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded}
        )#end sses.run()
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        categories_index,
        use_normalized_coordinates = True,
        min_score_thresh = MIN_ratio,#최소 인식률
        line_thickness = 2) #선두께
    
    
    frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('cam', frame)
    
    if key == ord("p"):
        Video_switch = False
    elif key == ord("s"):
        Video_switch = True;
        
    elif key == ord("q"):
        break


cv2.destroyAllWindows()
'''

while True:
    ret, frame = capture.read()
    key = cv2.waitKey(1) & 0xff

    frame_expanded = np.expand_dims(frame, axis = 0)

    #프레임 표시
    curtime = time.time()
    sec = curtime - prevtime
    prevtime = curtime
    fps = 1/ sec
    str = "FPS : %0.1f" % fps
    cv2.putText(frame, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
    #end 프레임
    
    (boxes, scores, classes, nums) = sses.run(#np.ndarray
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded}
        )#end sses.run()
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        categories_index,
        use_normalized_coordinates = True,
        min_score_thresh = MIN_ratio,#최소 인식률
        line_thickness = 2) #선두께

    '''
    try:
        print(car_info[0] )
        cv2.imshow('rrr', car_info[1])
    except:
        pass
    '''
    
    if not ret:
        break

    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            
            #frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('frame', frame)

            if key2 == ord('p'):
                break

    #frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('frame',frame)
    
    
    if key == ord("q"): 
        break

cv2.destroyAllWindows()

'''
    2 bicycle
    3 car
    4 motorcycle
    6 bus
    8 truck
'''
