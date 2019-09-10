import os
import cv2
import time
import numpy as np
import tensorflow as tf

from queue import Queue
from threading import Thread

from object_detection.utils import label_map_util as lmu
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

'''
큐 쓰레드 = 병렬처리
멀티프로세스 = 다른거 실행가능?!


'''
#define
time1 = time.time()
MIN_ratio = 0.8

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
#MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
LABEL_FILE = 'data/mscoco_label_map.pbtxt'
NUM_CLASSES = 90

label_map = lmu.load_labelmap(LABEL_FILE)
categories = lmu.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
categories_index = lmu.create_category_index(categories)
graph_file = MODEL_NAME + '/' + GRAPH_FILE_NAME
##end define
print("call label_map & categories : %0.5f" % (time.time() - time1))

def detect_objects(image_np, sess, detection_graph):
    time2 = time.time()
    #반복돌고있는 함수
    image_np_expanded = np.expand_dims(frame, axis = 0)
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    print("make tensor time : %0.5f" % (time.time() - time2))
    
    (boxes, scores, classes, nums) = sess.run( #np.ndarray real-time detection
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded}
        )#end sses.run()
    print("sess.run time : %0.5f" % (time.time() - time2))
    '''
    objects = [] #리스트 생성
    for index, value in enumerate(classes[0]):
        object_dict = {} #딕셔너리
        if scores[0][index] > MIN_ratio:
            object_dict[(categories_index.get(value)).get('name').encode('utf8')] = \
                    scores[0][index]
            objects.append(object_dict) #리스트 추가
    print(objects)
    '''
    

    '''
    
    vis_util.visualize_boxes_and_labels_on_image_array(#visualization
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        categories_index,
        use_normalized_coordinates = True,
        min_score_thresh = MIN_ratio,#최소 인식률
        line_thickness = 2)#선두께
    '''
def detect(input_Q, output_Q):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name = '')

        sess = tf.Session(graph = detection_graph)
    #적재는 한번
    while True:
        frame = input_Q.get()
        output_Q.put(detect_objects(frame, sess, detection_graph))

    sess.close()
    print("store in memoey time : %0.5f" % (time.time() - time1))
                                    
if __name__ == '__main__':
    prevtime = 0
    capture = cv2.VideoCapture(0)

    ob_input_q = Queue(3)
    ob_output_q = Queue()
    for i in range(1):
        ob_t = Thread(target=detect, args=(ob_input_q, ob_output_q))
        ob_t.daemon = True
        ob_t.start()
        print("Q thread start : %0.5f" % (time.time() - time1))

    
    
    #카메라 구간
    while True:
        ret, frame = capture.read()
        ob_input_q.put(frame)
        
        #프레임 표시
        curtime = time.time()
        sec = curtime - prevtime
        prevtime = curtime
        fps = 1/ sec
        str = "FPS : %0.1f" % fps
        cv2.putText(frame, str, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        #end 프레임
    
        cv2.imshow('cam', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
cv2.destroyAllWindows()
