# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A set of functions that are used for visualization.

These functions often receive an image, perform some visualization on the image.
The functions do not return a value, instead they modify the image itself.

"""
import cv2, re

import abc
import collections
# Set headless-friendly backend.
import matplotlib; matplotlib.use('Agg')  # pylint: disable=multiple-statements
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import six
import tensorflow as tf

import time
import pytesseract
from datetime import date, datetime

from object_detection.core import standard_fields as fields
from object_detection.utils import shape_utils


#from object_detection.utils.zzzz import Plate_char_set

global car_info
car_info = []
#end import global varibles


_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def _get_multiplier_for_color_randomness():
  """Returns a multiplier to get semi-random colors from successive indices.

  This function computes a prime number, p, in the range [2, 17] that:
  - is closest to len(STANDARD_COLORS) / 10
  - does not divide len(STANDARD_COLORS)

  If no prime numbers in that range satisfy the constraints, p is returned as 1.

  Once p is established, it can be used as a multiplier to select
  non-consecutive colors from STANDARD_COLORS:
  colors = [(p * i) % len(STANDARD_COLORS) for i in range(20)]
  """
  num_colors = len(STANDARD_COLORS)
  prime_candidates = [5, 7, 11, 13, 17]

  # Remove all prime candidates that divide the number of colors.
  prime_candidates = [p for p in prime_candidates if num_colors % p]
  if not prime_candidates:
    return 1

  # Return the closest prime number to num_colors / 10.
  abs_distance = [np.abs(num_colors / 10. - p) for p in prime_candidates]
  num_candidates = len(abs_distance)
  inds = [i for _, i in sorted(zip(abs_distance, range(num_candidates)))]
  return prime_candidates[inds[0]]


def save_image_array_as_png(image, output_path):
  """Saves an image (represented as a numpy array) to PNG.

  Args:
    image: a numpy array with shape [height, width, 3].
    output_path: path to which image should be written.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  with tf.gfile.Open(output_path, 'w') as fid:
    image_pil.save(fid, 'PNG')


def encode_image_array_as_png_str(image):
  """Encodes a numpy array into a PNG string.

  Args:
    image: a numpy array with shape [height, width, 3].

  Returns:
    PNG encoded image string.
  """
  image_pil = Image.fromarray(np.uint8(image))
  output = six.BytesIO()
  image_pil.save(output, format='PNG')
  png_string = output.getvalue()
  output.close()
  return png_string


def draw_bounding_box_on_image_array(image,
                                     ymin,
                                     xmin,
                                     ymax,
                                     xmax,
                                     color='red',
                                     thickness=4,
                                     display_str_list=(),
                                     use_normalized_coordinates=True):
  """Adds a bounding box to an image (numpy array).

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.
  
  이미지에 경계 상자 추가(넘파이 배열).

  경계 상자 좌표는 절대값 (픽셀) 또는
  use_normalized_coordinates 인수를 설정하여 정규화 된 좌표 반환
  
  Args:
    image: a numpy array with shape [height, width, 3].
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  #픽셀마다 int8형을 주고 깔라로 변환함
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB') #전체 이미지 픽셀 형변환
  draw_bounding_box_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                             thickness, display_str_list,
                             use_normalized_coordinates)

  
  
  np.copyto(image, np.array(image_pil))
  #print("image_pil:" + str(image_pil))
  #print("image:" + str(image))
#end def draw_bounding_box_on_image_array


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
  """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
  
  draw = ImageDraw.Draw(image)
  # print(type(draw)) <class 'PIL.ImageDraw.ImageDraw'>
  # print(draw) <PIL.ImageDraw.ImageDraw object at 0x000001FAC2BA9550> 주소값은 이미지
  #박스 그리는데 사이즈 주는거 가로세로 폭 가져옴
  im_width, im_height = image.size
  
  if use_normalized_coordinates:
    (left,  right, top, bottom) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
    
    #car_recognition = image[top:bottom, left:right]
  '''
  else:
    area = (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    #car_recognition = image[top:bottom, left:right]
  '''
    
  '''원형코드
  #if im_width*0.6 <= left
  draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
  
  #cr_image = image.crop((left, top, right, bottom))
  #cr_image.show()
  
  try:
    font = ImageFont.truetype('arial.ttf', 24)
  except IOError:
    font = ImageFont.load_default() 

  # If the total height of the display strings added to the top of the bounding
  # box exceeds the top of the image, stack the strings below the bounding box
  # instead of above.
  display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

  # Each display_str has a top and bottom margin of 0.05x.
  total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

  if top > total_display_str_height:
    text_bottom = top
  else:
    text_bottom = bottom + total_display_str_height
    
  # Reverse list and print from bottom to top.
  for display_str in display_str_list[::-1]: #이미지 리스트 반복
    # print(type(display_str) ) type = str
    #print(display_str + "   가로폭 : " + str(right - left) )
    if im_width*0.6 <= right-left:
      print(display_str + "   가로폭 : " + str(right - left) )
      
      text_width, text_height = font.getsize(display_str)
      
      margin = np.ceil(0.05 * text_height)
      draw.rectangle( [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)

      draw.text( (left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
      text_bottom -= text_height - 2 * margin
  '''
  
  if im_height*0.47 <= bottom-top and im_height*0.71 >= bottom-top:
    if im_width*0.33 <= right-left and im_width*0.57 >= right-left:
      draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)], width=thickness, fill=color)
      
      #cr_image = image.crop((left, top, right, bottom))
      #cr_image.show()
      
      try:
        font = ImageFont.truetype('arial.ttf', 24)
      except IOError:
        font = ImageFont.load_default() 

      # If the total height of the display strings added to the top of the bounding
      # box exceeds the top of the image, stack the strings below the bounding box
      # instead of above.
      display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]

      # Each display_str has a top and bottom margin of 0.05x.
      total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

      
      if top > total_display_str_height:
        text_bottom = top
      
      else:
        text_bottom = bottom + total_display_str_height
      
      # Reverse list and print from bottom to top.
      for display_str in display_str_list[::-1]: #이미지 리스트 반복
        # print(type(display_str) ) type = str
        #print(display_str + "   가로폭 : " + str(right - left) )
        #if im_heigh*0.65 <= bottom-top and im_width*0.72 >= bottom-top:
          #print(display_str + "   가로폭 : " + str(right - left) )
        if display_str[:2] == 'tv':
          break
        
        else:
          
        
          print("%s - 가로폭: %0.5f 세로폭: %0.5f" %(display_str, (right-left), (bottom-top)) )
            
          cr_image = image.crop((left+50, top, right-50, bottom) )
            
          cr_image.save("car_image/" + what_now_time() + str(display_str[:3]) + str(int(bottom-top)) + ".jpg")
          #cr_image.save("car_image/" + str(int(right-left)) + str(display_str[:2]) + str(int(bottom-top)) + ".jpg")

          cr_image = image.crop((left+120, top+200, right-100, bottom) )
          #cv2.imshow('cut', ocv)
          #return cr_image
          
          #global plate_chars
          #global plate_image
          try:
            plate_chars, plate_image = number_recognition(cr_image)       #return 번호판이름
            car_info.append(plate_chars)
            car_info.append(plate_image)
            '''
            if car_info[0] == None:
              car_info.append(plate_chars)
              car_info.append(plate_image)
            else:
              car_info[0] = plate_chars
              car_info[1] = plate_image
            '''
            if car_info[0] != None:
              car_info[0] = plate_chars
              car_info[1] = plate_image
                
            #print("저장한새끼 : " + str(car_info) )
            cr_image.save("car_image/" + "pLate-" + str(plate_chars) + ".jpg")
          except:
            print("응지나가")
            
          text_width, text_height = font.getsize(display_str)
          
          margin = np.ceil(0.05 * text_height)
          draw.rectangle( [(left, text_bottom - text_height - 2 * margin), (left + text_width, text_bottom)], fill=color)

          draw.text( (left + margin, text_bottom - text_height - margin), display_str, fill='black', font=font)
          text_bottom -= text_height - 2 * margin

def what_now_time():
  day = datetime.now()
  return str('{0.year:04}{0.month:02}{0.day:02}_{0.hour:02}h{0.minute:02}m{0.second:02}s'.format(day) )


def number_recognition(cut_image):
    # --Read Input Image-- (이미지 불러오기)
    #src = cv2.imread(cut_image) #이미지 불러오기


    src = cv2.cvtColor(np.array(cut_image), cv2.COLOR_RGB2BGR)
    
    print("불러오기는 했음")
    '''
    dst = src.copy()  # 이미지영역을 반으로 자르기(번호판 인식률 속도를 높이기 위함)
    dst = src[480:960, 50:670]
    cv2.imshow("half img", dst)
    cv2.waitKey(0)
    '''

    prevtime = time.time()  # 걸린 시간 체크하는 함수

    # 변수 선언
    height, width, channel = src.shape  # 이미지에 대한 값을 가질 변수

    numcheck = 0  # 반복문에서 번호판 문자열 검사할 변수
    charsok = 0  # 반복문에서 번호판 글자를 제대로 읽었는지 검사할 변수
    add_w_padding, add_h_padding = 0, 0  # 추가할 padding값을 가질 변수
    w_padding_max, h_padding_max = 0, 0  # 일정한 padding값을 가지게되었을때 반복문을 제어할 변수

    # --Convert Image to Grayscale-- (이미지 흑백변환)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # 이미지 흑백변환

    # --Maximize Contrast(Optional)-- (흑백대비 최대화)

    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    # --Adaptive Thresholding-- (가우시안블러(이미지 노이즈 제거) 및 쓰레시 홀딩)

    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)  # GaussianBlur 적용

    img_thresh = cv2.adaptiveThreshold(  # adaptiveThreshold 적용
        img_blurred,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    # --Find Contours-- (윤곽선 찾기)

    contours, hierarchy = cv2.findContours(  # opencv의 findContours를 이용하여 contours에 저장
        img_thresh,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)  # numpy.zeros를 이용하여 윤곽선 범위 저장

    cv2.drawContours(temp_result, contours, -1, (255, 255, 255))  # 윤곽선 그리기

    # --Prepare Data-- (데이터 비교하기, 글자영역으로 추정되는 rectangle 그리기)

    temp_result = np.zeros((height, width, channel),
                           dtype=np.uint8)  # drawContours를 이용해 그린 윤곽선에 다시 numpy.zeros를 이용해 다시 윤곽선 범위 저장 (안하면 윤곽선 좀 남아있음)

    contours_dict = []  # contour 정보를 모두 저장받을 리스트 변수

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # 위치 높낮이 데이터 정보 저장
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255),
                      thickness=2)  # 윤곽선을 감싸는 사각형 구하기

        # insert to dict
        contours_dict.append({  # contour 정보를 모두 저장
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })
        
    # --Select Candidates by Char Size-- (글자 같은 영역 찾기)

    MIN_AREA = 80  # 윤곽선의 가운데 렉트 최소 넓이 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 바운드 렉트의 최소 너비와 높이는 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 바운드 렉트의 비율 가로 대비 세로 비율 최솟값 0.25, 최댓값 1.0

    possible_contours = []  # 글자로 예상되는 contour들을 저장받을 리스트 변수

    cnt = 0  # count 변수
    for d in contours_dict:  # contours_dict에 저장된 것을 조건에 맞다면 possible_contours에 append
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']

        if area > MIN_AREA \
                and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        #     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)  # 글자로 예상되는 영역만 rectangle 그리기

    # --Select Candidates by Arrangement of Contours-- (글자의 연속성(번호판으로 예상되는 영역) 찾기)

    MAX_DIAG_MULTIPLYER = 4.7  # 5 contour와 contour의 사이의 길이 (값계속 바꿔가면서 테스트 해야함)
    MAX_ANGLE_DIFF = 13  # 12.0 첫번째 contour와 두번째 contour의 직각 삼각형의 앵글 세타각도
    MAX_AREA_DIFF = 0.5  # 0.5  면적의 차이
    MAX_WIDTH_DIFF = 0.8  # 0.8 contour 간의 가로길이 차이
    MAX_HEIGHT_DIFF = 0.2  # 0.2 contour 간의 세로길이 차이
    MIN_N_MATCHED = 4  # 3 글자영역으로 예측된 것의 최소 갯수 (ex 3개이상이면 번호판일 것)

    def find_chars(contour_list):  # 재귀함수로 번호판 후보군을 계속 찾음
        matched_result_idx = []  # 최종 결과값의 인덱스를 저장

        for d1 in contour_list:  # 컨투어(d1, d2)를 서로 비교
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(
                    np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))  # d1과 d2거리를 계산
                if dx == 0:  # dx의 절댓값이 0이라면 (d1과 d2의 x값을 갖고 있다면)
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))  # 아크탄젠트 값을 구함 (라디안)
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])  # 면적의 비율
                width_diff = abs(d1['w'] - d2['w']) / d1['w']  # 너비의 비율
                height_diff = abs(d1['h'] - d2['h']) / d1['h']  # 높이의 비율

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                        and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                        and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])  # 설정한 파라미터 기준에 맞는 값들의 인덱스만 append

            # append this contour
            matched_contours_idx.append(d1['idx'])  # d1을 빼먹고 넣었으므로 d1도 넣어줌

            if len(matched_contours_idx) < MIN_N_MATCHED:  # 예상한 번호판의 최소 갯수가 맞지 않다면 continue
                continue

            matched_result_idx.append(matched_contours_idx)  # 최종후보군으로 넣음 append

            unmatched_contour_idx = []  # 최종 후보군이 아닌 것들도 아닌 것들끼리 한번 더 비교
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:  # matched_contour_idx가 아닌 것들
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours,
                                        unmatched_contour_idx)  # numpy.take를 이용해서 unmathced_contour에 저장

            # recursive
            recursive_contour_list = find_chars(unmatched_contour)  # 다시 돌려봄

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)  # 최종 결과값을 mathced_result_idx에 다시 저장

            break

        return matched_result_idx

    result_idx = find_chars(possible_contours)

    matched_result = []  # 예상되는 번호판 contour정보를 담을 리스트 변수
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours (번호판 contour 그리기)
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:  # 번호판으로 예상되는 역역을 그림
        for d in r:
            #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']),
                          color=(255, 255, 255),
                          thickness=2)

    # --Rotate Plate Images-- (이미지 회전)

    plate_imgs = []  # 번호판 이미지를 담을 리스트 변수
    plate_infos = []  # 번호판 정보를 담을 리스트 변수

    longest_idx, longest_text = -1, 0  # idx값 초기화
    plate_chars = []  # 번호판 리스트 변수

    while charsok == 0:  # 번호판 글자로 예상되는 값이 나올 때까지 반복
        PLATE_WIDTH_PADDING = 1.2 + add_w_padding  # 가로 패딩 값 예제 디폴트는 1.3
        PLATE_HEIGHT_PADDING = 1.51 + add_h_padding  # 세로 패딩 값 예제 디폴트는 1.5
        MIN_PLATE_RATIO = 3  # 3 최소 번호판 비율
        MAX_PLATE_RATIO = 10  # 10 최대 번호판 비율

        for i, matched_chars in enumerate(matched_result):
            sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

            plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
            plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

            plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

            sum_height = 0
            for d in sorted_chars:
                sum_height += d['h']

            plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

            triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']  # 번호판의 간격을 삼각형을 기준으로 세타 값을 구함
            triangle_hypotenus = np.linalg.norm(
                np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
                np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
            )

            angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))  # 라디안 값을 구해서 각도로 바꿈

            rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle,
                                                      scale=1.0)  # 로테이션 이미지 구하기

            img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))  # 이미지 변형

            img_cropped = cv2.getRectSubPix(  # 회전된 이미지에서 원하는 부분만 자름
                img_rotated,
                patchSize=(int(plate_width), int(plate_height)),
                center=(int(plate_cx), int(plate_cy))
            )

            if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / \
                    img_cropped.shape[
                        0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:  # 번호판 비율이 맞지 않다면 continue
                continue

            plate_imgs.append(img_cropped)  # plate_imgs에 append

            plate_infos.append({  # plate_infos에 append
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height)
            })

        # --Another Thresholding to Find Chars-- (찾은문자에서 다시 쓰레시홀딩)

        for i, plate_img in enumerate(plate_imgs):
            if numcheck > 3:  # 예상되는 번호판 영역에서 문자열을 검사해 숫자 3개가 넘는다면(번호판일 확률이 높다면)
                break

            plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
            _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0,
                                         type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 쓰레시홀딩

            # find contours again (same as above)
            contours, hierarchy = cv2.findContours(plate_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # contour 다시 찾기

            plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
            plate_max_x, plate_max_y = 0, 0

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)  # for문을 돌려 boundingRect를 다시 구함

                area = w * h  # 면적
                ratio = w / h  # 비율

                if area > MIN_AREA \
                        and w > MIN_WIDTH and h > MIN_HEIGHT \
                        and MIN_RATIO < ratio < MAX_RATIO:  # 설정한 기준(파라미터)에 맞는지 다시 확인
                    if x < plate_min_x:  # x, y의 최댓값,최소값을 구함
                        plate_min_x = x
                    if y < plate_min_y:
                        plate_min_y = y
                    if x + w > plate_max_x:
                        plate_max_x = x + w
                    if y + h > plate_max_y:
                        plate_max_y = y + h

            img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]  # 이미지를 번호판 부분만 잘라내기

            img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)  # GaussianBlur(노이즈 제거)
            _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0,
                                          type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 쓰레시홀딩 한번 더
            img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10,
                                            borderType=cv2.BORDER_CONSTANT,  # 이미지에 패딩(여백)을 줌
                                            value=(0, 0, 0))  # 검은색

            cv2.imwrite('00.jpg', img_result)
            chars = pytesseract.image_to_string(Image.open('00.jpg'), config='--psm 7 --oem 0',
                                                lang='kor')  # 저장한 이미지를 불러 pytesseract로 읽음
            nowtime = time.time()
            #print("이미지 불러 온 후 글자 : " + chars)

            result_chars = ''  # 번호판 인식 문자 정보를 담을 변수
            has_digit = False

            
            for c in chars:  # 판독해서 특수문자를 제외한 한글 문자와 숫자 넣기
                if ord('가') <= ord(c) <= ord('힣') or c.isdigit():  # 숫자나 한글이 포함되어 있는지
                    if c.isdigit():
                        has_digit = True  # 숫자가 하나라도 있는지
                    result_chars += c

            for n in range(len(result_chars)):  # 번호판 형식이 맞는지 다시한번 검사 및 문자열 자르기
                if len(result_chars) < 7:  # 번호판 길이가 7자리(번호판의 최소 길이는 7자리)보다 짧다면
                    break
                elif result_chars[0].isdigit() == False:  # 첫문자가 문자라면(숫자가 아니라면) 자르기
                    result_chars = result_chars[1:result_chars.__len__()]

                elif result_chars[len(result_chars) - 1].isdigit() == False:  # 마지막 문자가 한글데이터라면(숫자가 아니라면) 자르기
                    result_chars = result_chars[0:(result_chars.__len__() - 1)]

            plate_chars.append(result_chars)  # 결과 result_chars를 plate_chars에 append

            for j in range(len(result_chars)):  # 번호판의 배열이 나오는지를 검사 ex) 12가3456(7자리번호판) or 123가4567(8자리번호판)
                if len(result_chars) < 7:  # 결과길이가 7자리(번호판의 최소 길이는 7자리)보다 짧다면
                    break
                elif (j == 2 and result_chars[j].isdigit() == True) and result_chars[
                    j + 1].isdigit() == True:  # 번호판의 3번째와 4번째가 동시에 숫자라면(글자가 아니라면)
                    break
                elif (j != 2 and j != 3) and result_chars[j].isdigit() == False:  # 번호판의 3,4번째(글자영역)가 아닌데 문자라면
                    break
                elif (j == 2 and result_chars[j].isdigit() == False) and result_chars[
                    j + 1].isdigit() == False:  # 번호판의 3,4번째자리가 둘 다 문자라면
                    break
                if 6 <= j and result_chars[j].isdigit() == True:  # 6번째까지 숫자자리에 문자가 없고 7번째 영역이 숫자라면 번호판일 것
                    charsok = 1  # 반복문을 멈춤
                    break

            if has_digit and len(result_chars) > longest_text:  # 조건을 만족하면
                longest_idx = i  # 가장 긴 값을 인덱스로 줌

            for numch, in result_chars:  # 문자열 검사를 통해 숫자가 3개 이상이라면 번호판일 확률이 높으므로 이 plate_imgs는 번호판일 것임 그러므로 패딩값을 조절하면 되기에 이미지는 고정할 것
                if numch.isdigit() == True:
                    numcheck += 1

        # --Result-- (결과값)

        info = plate_infos[longest_idx]  # 번호판 좌표 정보 담기
        chars = plate_chars[longest_idx]  # 번호판 문자열 정보 담기
        
        # 가로 패딩값을 0.1씩 늘림 -> 가로를 초기화 후 세로 패딩값을 0.1씩 늘림 -> 가로 세로 패딩값을 0.1씩 늘림 모두 0.6이 되면 프로그램 종료
        if add_w_padding <= 0.6 and w_padding_max == 0:  # w패딩이 0.5보다 작다면 (가로 패딩만 먼저 늘려보기)
            add_w_padding += 0.1  # w패딩을 0.1씩 증가

        elif w_padding_max == 1 and add_h_padding <= 0.6 and h_padding_max == 0:  # w패딩이 0.5를 찍고 h패딩이 0.5보다 작다면
            add_w_padding = 0  # w패딩을 다시 Default값으로 (세로 패딩만 늘려보기)
            add_h_padding += 0.1  # h패딩을 0.1씩 증가

        if add_w_padding == 0.6:  # 0.6까지 늘어났다면
            w_padding_max = 1
        if add_h_padding == 0.6:  # 0.6까지 늘어났다면
            h_padding_max = 1
            add_w_padding = 0
            add_h_padding = 0

        if w_padding_max == 1 and h_padding_max == 1:  # 너비높이 0.1씩 증가시키기
            add_w_padding += 0.1
            add_h_padding += 0.1
            if add_w_padding == 0.6 and add_h_padding == 0.6:  # 패딩값을 너비 높이 다 0.6씩 늘렸다면(번호판을 못 찾았다면)
                break
        # 초기화
        numcheck = 0
        plate_imgs = []
        plate_chars = []

    sec = nowtime - prevtime

    recc = fix_plate_char_set(chars)
    print("걸린시간 %0.5f" % sec)
    print("최종 값 : " + recc)

    img_out = src.copy()
    cv2.rectangle(img_out, pt1=(info['x'], info['y']), pt2=(info['x'] + info['w'], info['y'] + info['h']),
                    color=(255, 0, 0), thickness=2) # 원본 이미지에 번호판 영역 그리기

    cv2.imwrite('result.jpg', img_out) #원본 이미지에서 번호판 영역 추출한 사진
      
    return recc, img_result # 결과값 return

  
def fix_plate_char_set(txt):
  HAN_CHO = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']


  HAN_JUNG = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
             'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
             'ㅣ']


  HAN_JONG = ['  ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
             'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
             'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

  retxt = ''
  for c in txt:
      if ord('가') <= ord(c) <= ord('힣'): #한글 부시기
          cc = ord(c) - 44032     # 한글 완성자의 유니코드 포인터 값 추출
          cho = cc // (21 * 28)   # 초성 값 추출
          jung = (cc // 28) % 21  # 중성 값 추출
          jong = cc % 28          # 종성 값 추출
            
          if jung == 17:
              jung = jung-4
          if jung == 12:
              jung = jung-4
          if jung == 6: # ㅕ
              jung = jung-2 # ㅓ
          if jung == 2: #ㅑ
              jung = jung-2# ㅏ
                
          jong = 0 #종값 삭제

          car_val = ((cho*21)+jung)*28+jong+0xAC00 #문자열 합쳐서 붙여버릴거
          retxt += chr(car_val)
        
      elif ord('?') == ord(c): #? 9를 오인함 그래서 바꿔버릴거 
          c = str(9)
          retxt += c

      elif ord('0') <= ord(c) <= ord('9'): #정상 숫자면 저장
          retxt += str(c)
      #end for txt
            
  return retxt

def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
  """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  image_pil = Image.fromarray(image)
  draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                               display_str_list_list)
  np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=4,
                                 display_str_list_list=()):
  """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
  boxes_shape = boxes.shape
  if not boxes_shape:
    return
  
  if len(boxes_shape) != 2 or boxes_shape[1] != 4:
    raise ValueError('Input must be of size [N, 4]')
  
  for i in range(boxes_shape[0]):
    display_str_list = ()
    if display_str_list_list:
      display_str_list = display_str_list_list[i]
      
    draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                               boxes[i, 3], color, thickness, display_str_list)


def create_visualization_fn(category_index, include_masks=False,
                            include_keypoints=False, include_track_ids=False,
                            **kwargs):
  """Constructs a visualization function that can be wrapped in a py_func.

  py_funcs only accept positional arguments. This function returns a suitable
  function with the correct positional argument mapping. The positional
  arguments in order are:
  0: image
  1: boxes
  2: classes
  3: scores
  [4-6]: masks (optional)
  [4-6]: keypoints (optional)
  [4-6]: track_ids (optional)

  -- Example 1 --
  vis_only_masks_fn = create_visualization_fn(category_index,
    include_masks=True, include_keypoints=False, include_track_ids=False,
    **kwargs)
  image = tf.py_func(vis_only_masks_fn,
                     inp=[image, boxes, classes, scores, masks],
                     Tout=tf.uint8)

  -- Example 2 --
  vis_masks_and_track_ids_fn = create_visualization_fn(category_index,
    include_masks=True, include_keypoints=False, include_track_ids=True,
    **kwargs)
  image = tf.py_func(vis_masks_and_track_ids_fn,
                     inp=[image, boxes, classes, scores, masks, track_ids],
                     Tout=tf.uint8)

  Args:
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    include_masks: Whether masks should be expected as a positional argument in
      the returned function.
    include_keypoints: Whether keypoints should be expected as a positional
      argument in the returned function.
    include_track_ids: Whether track ids should be expected as a positional
      argument in the returned function.
    **kwargs: Additional kwargs that will be passed to
      visualize_boxes_and_labels_on_image_array.

  Returns:
    Returns a function that only takes tensors as positional arguments.
  """

  def visualization_py_func_fn(*args):
    """Visualization function that can be wrapped in a tf.py_func.

    Args:
      *args: First 4 positional arguments must be:
        image - uint8 numpy array with shape (img_height, img_width, 3).
        boxes - a numpy array of shape [N, 4].
        classes - a numpy array of shape [N].
        scores - a numpy array of shape [N] or None.
        -- Optional positional arguments --
        instance_masks - a numpy array of shape [N, image_height, image_width].
        keypoints - a numpy array of shape [N, num_keypoints, 2].
        track_ids - a numpy array of shape [N] with unique track ids.

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3) with overlaid
      boxes.
    """
    image = args[0]
    boxes = args[1]
    classes = args[2]
    scores = args[3]
    masks = keypoints = track_ids = None
    pos_arg_ptr = 4  # Positional argument for first optional tensor (masks).
    if include_masks:
      masks = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_keypoints:
      keypoints = args[pos_arg_ptr]
      pos_arg_ptr += 1
    if include_track_ids:
      track_ids = args[pos_arg_ptr]

    return visualize_boxes_and_labels_on_image_array(
        image,
        boxes,
        classes,
        scores,
        category_index=category_index,
        instance_masks=masks,
        keypoints=keypoints,
        track_ids=track_ids,
        **kwargs)
  return visualization_py_func_fn


def _resize_original_image(image, image_shape):
  image = tf.expand_dims(image, 0)
  image = tf.image.resize_images(
      image,
      image_shape,
      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      align_corners=True)
  return tf.cast(tf.squeeze(image, 0), tf.uint8)


def draw_bounding_boxes_on_image_tensors(images,
                                         boxes,
                                         classes,
                                         scores,
                                         category_index,
                                         original_image_spatial_shape=None,
                                         true_image_shape=None,
                                         instance_masks=None,
                                         keypoints=None,
                                         track_ids=None,
                                         max_boxes_to_draw=20,
                                         min_score_thresh=0.2,
                                         use_normalized_coordinates=True):
  """Draws bounding boxes, masks, and keypoints on batch of image tensors.

  Args:
    images: A 4D uint8 image tensor of shape [N, H, W, C]. If C > 3, additional
      channels will be ignored. If C = 1, then we convert the images to RGB
      images.
    boxes: [N, max_detections, 4] float32 tensor of detection boxes.
    classes: [N, max_detections] int tensor of detection classes. Note that
      classes are 1-indexed.
    scores: [N, max_detections] float32 tensor of detection scores.
    category_index: a dict that maps integer ids to category dicts. e.g.
      {1: {1: 'dog'}, 2: {2: 'cat'}, ...}
    original_image_spatial_shape: [N, 2] tensor containing the spatial size of
      the original image.
    true_image_shape: [N, 3] tensor containing the spatial size of unpadded
      original_image.
    instance_masks: A 4D uint8 tensor of shape [N, max_detection, H, W] with
      instance masks.
    keypoints: A 4D float32 tensor of shape [N, max_detection, num_keypoints, 2]
      with keypoints.
    track_ids: [N, max_detections] int32 tensor of unique tracks ids (i.e.
      instance ids for each object). If provided, the color-coding of boxes is
      dictated by these ids, and not classes.
    max_boxes_to_draw: Maximum number of boxes to draw on an image. Default 20.
    min_score_thresh: Minimum score threshold for visualization. Default 0.2.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    4D image tensor of type uint8, with boxes drawn on top.
  """
  # Additional channels are being ignored.
  if images.shape[3] > 3:
    images = images[:, :, :, 0:3]
  elif images.shape[3] == 1:
    images = tf.image.grayscale_to_rgb(images)
  visualization_keyword_args = {
      'use_normalized_coordinates': use_normalized_coordinates,
      'max_boxes_to_draw': max_boxes_to_draw,
      'min_score_thresh': min_score_thresh,
      'agnostic_mode': False,
      'line_thickness': 4
  }
  if true_image_shape is None:
    true_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 3])
  else:
    true_shapes = true_image_shape
  if original_image_spatial_shape is None:
    original_shapes = tf.constant(-1, shape=[images.shape.as_list()[0], 2])
  else:
    original_shapes = original_image_spatial_shape

  visualize_boxes_fn = create_visualization_fn(
      category_index,
      include_masks=instance_masks is not None,
      include_keypoints=keypoints is not None,
      include_track_ids=track_ids is not None,
      **visualization_keyword_args)

  elems = [true_shapes, original_shapes, images, boxes, classes, scores]
  if instance_masks is not None:
    elems.append(instance_masks)
  if keypoints is not None:
    elems.append(keypoints)
  if track_ids is not None:
    elems.append(track_ids)

  def draw_boxes(image_and_detections):
    """Draws boxes on image."""
    true_shape = image_and_detections[0]
    original_shape = image_and_detections[1]
    if true_image_shape is not None:
      image = shape_utils.pad_or_clip_nd(image_and_detections[2],
                                         [true_shape[0], true_shape[1], 3])
    if original_image_spatial_shape is not None:
      image_and_detections[2] = _resize_original_image(image, original_shape)

    image_with_boxes = tf.py_func(visualize_boxes_fn, image_and_detections[2:],
                                  tf.uint8)
    return image_with_boxes

  images = tf.map_fn(draw_boxes, elems, dtype=tf.uint8, back_prop=False)
  return images


def draw_side_by_side_evaluation_image(eval_dict,
                                       category_index,
                                       max_boxes_to_draw=20,
                                       min_score_thresh=0.2,
                                       use_normalized_coordinates=True):
  """Creates a side-by-side image with detections and groundtruth.

  Bounding boxes (and instance masks, if available) are visualized on both
  subimages.

  Args:
    eval_dict: The evaluation dictionary returned by
      eval_util.result_dict_for_batched_example() or
      eval_util.result_dict_for_single_example().
    category_index: A category index (dictionary) produced from a labelmap.
    max_boxes_to_draw: The maximum number of boxes to draw for detections.
    min_score_thresh: The minimum score threshold for showing detections.
    use_normalized_coordinates: Whether to assume boxes and kepoints are in
      normalized coordinates (as opposed to absolute coordiantes).
      Default is True.

  Returns:
    A list of [1, H, 2 * W, C] uint8 tensor. The subimage on the left
      corresponds to detections, while the subimage on the right corresponds to
      groundtruth.
  """
  detection_fields = fields.DetectionResultFields()
  input_data_fields = fields.InputDataFields()

  images_with_detections_list = []

  # Add the batch dimension if the eval_dict is for single example.
  if len(eval_dict[detection_fields.detection_classes].shape) == 1:
    for key in eval_dict:
      if key != input_data_fields.original_image:
        eval_dict[key] = tf.expand_dims(eval_dict[key], 0)

  for indx in range(eval_dict[input_data_fields.original_image].shape[0]):
    instance_masks = None
    
    if detection_fields.detection_masks in eval_dict:
      instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[detection_fields.detection_masks][indx], axis=0),
          tf.uint8)
    keypoints = None
    if detection_fields.detection_keypoints in eval_dict:
      keypoints = tf.expand_dims(
          eval_dict[detection_fields.detection_keypoints][indx], axis=0)
    groundtruth_instance_masks = None
    if input_data_fields.groundtruth_instance_masks in eval_dict:
      groundtruth_instance_masks = tf.cast(
          tf.expand_dims(
              eval_dict[input_data_fields.groundtruth_instance_masks][indx],
              axis=0), tf.uint8)

    images_with_detections = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_classes][indx], axis=0),
        tf.expand_dims(
            eval_dict[detection_fields.detection_scores][indx], axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=instance_masks,
        keypoints=keypoints,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates)
    
    images_with_groundtruth = draw_bounding_boxes_on_image_tensors(
        tf.expand_dims(
            eval_dict[input_data_fields.original_image][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_boxes][indx], axis=0),
        tf.expand_dims(
            eval_dict[input_data_fields.groundtruth_classes][indx], axis=0),
        tf.expand_dims(
            tf.ones_like(
                eval_dict[input_data_fields.groundtruth_classes][indx],
                dtype=tf.float32),
            axis=0),
        category_index,
        original_image_spatial_shape=tf.expand_dims(
            eval_dict[input_data_fields.original_image_spatial_shape][indx],
            axis=0),
        true_image_shape=tf.expand_dims(
            eval_dict[input_data_fields.true_image_shape][indx], axis=0),
        instance_masks=groundtruth_instance_masks,
        keypoints=None,
        max_boxes_to_draw=None,
        min_score_thresh=0.0,
        use_normalized_coordinates=use_normalized_coordinates)
    images_with_detections_list.append(
        tf.concat([images_with_detections, images_with_groundtruth], axis=2))
    #print(indx)
    #end for indx
    
  return images_with_detections_list


def draw_keypoints_on_image_array(image,
                                  keypoints,
                                  color='red',
                                  radius=2,
                                  use_normalized_coordinates=True):
  """Draws keypoints on an image (numpy array).

  Args:
    image: a numpy array with shape [height, width, 3].
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
  draw_keypoints_on_image(image_pil, keypoints, color, radius,
                          use_normalized_coordinates)
  np.copyto(image, np.array(image_pil))


def draw_keypoints_on_image(image,
                            keypoints,
                            color='red',
                            radius=2,
                            use_normalized_coordinates=True):
  """Draws keypoints on an image.

  Args:
    image: a PIL.Image object.
    keypoints: a numpy array with shape [num_keypoints, 2].
    color: color to draw the keypoints with. Default is red.
    radius: keypoint radius. Default value is 2.
    use_normalized_coordinates: if True (default), treat keypoint values as
      relative to the image.  Otherwise treat them as absolute.
  """
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  keypoints_x = [k[1] for k in keypoints]
  keypoints_y = [k[0] for k in keypoints]
  if use_normalized_coordinates:
    keypoints_x = tuple([im_width * x for x in keypoints_x])
    keypoints_y = tuple([im_height * y for y in keypoints_y])
  for keypoint_x, keypoint_y in zip(keypoints_x, keypoints_y):
    draw.ellipse([(keypoint_x - radius, keypoint_y - radius),
                  (keypoint_x + radius, keypoint_y + radius)],
                 outline=color, fill=color)


def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
  """Draws mask on an image.

  Args:
    image: uint8 numpy array with shape (img_height, img_height, 3)
    mask: a uint8 numpy array of shape (img_height, img_height) with
      values between either 0 or 1.
    color: color to draw the keypoints with. Default is red.
    alpha: transparency value between 0 and 1. (default: 0.4)

  Raises:
    ValueError: On incorrect data type for image or masks.
  """
  if image.dtype != np.uint8:
    raise ValueError('`image` not of type np.uint8')
  if mask.dtype != np.uint8:
    raise ValueError('`mask` not of type np.uint8')
  if np.any(np.logical_and(mask != 1, mask != 0)):
    raise ValueError('`mask` elements should be in [0, 1]')
  if image.shape[:2] != mask.shape:
    raise ValueError('The image has spatial dimensions %s but the mask has '
                     'dimensions %s' % (image.shape[:2], mask.shape))
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(image)

  solid_color = np.expand_dims(
      np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
  pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  np.copyto(image, np.array(pil_image.convert('RGB')))

def visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes,
    scores,
    category_index,
    instance_masks=None,
    instance_boundaries=None,
    keypoints=None,
    track_ids=None,
    use_normalized_coordinates=False,
    max_boxes_to_draw=20,
    min_score_thresh=.5,
    agnostic_mode=False,
    line_thickness=4,
    groundtruth_box_visualization_color='black',
    skip_scores=False,
    skip_labels=False,
    skip_track_ids=False):
  """Overlay labeled boxes on an image with formatted scores and label names.

  This function groups boxes that correspond to the same location
  and creates a display string for each detection and overlays these
  on the image. Note that this function modifies the image in place, and returns
  that same image.

  Args:
    image: uint8 numpy array with shape (img_height, img_width, 3)
    boxes: a numpy array of shape [N, 4]
    classes: a numpy array of shape [N]. Note that class indices are 1-based,
      and match the keys in the label map.
    scores: a numpy array of shape [N] or None.  If scores=None, then
      this function assumes that the boxes to be plotted are groundtruth
      boxes and plot all boxes as black with no classes or scores.
    category_index: a dict containing category dictionaries (each holding
      category index `id` and category name `name`) keyed by category indices.
    instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
    instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
    keypoints: a numpy array of shape [N, num_keypoints, 2], can
      be None
    track_ids: a numpy array of shape [N] with unique track ids. If provided,
      color-coding of boxes will be determined by these ids, and not the class
      indices.
    use_normalized_coordinates: whether boxes is to be interpreted as
      normalized coordinates or not.
    max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
      all boxes.
    min_score_thresh: minimum score threshold for a box to be visualized
    agnostic_mode: boolean (default: False) controlling whether to evaluate in
      class-agnostic mode or not.  This mode will display scores but ignore
      classes.
    line_thickness: integer (default: 4) controlling line width of the boxes.
    groundtruth_box_visualization_color: box color for visualizing groundtruth
      boxes
    skip_scores: whether to skip score when drawing a single detection
    skip_labels: whether to skip label when drawing a single detection
    skip_track_ids: whether to skip track id when drawing a single detection

  Returns:
    uint8 numpy array with shape (img_height, img_width, 3) with overlaid boxes.
  """
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
  box_to_display_str_map = collections.defaultdict(list)
  box_to_color_map = collections.defaultdict(str)
  box_to_instance_masks_map = {}
  box_to_instance_boundaries_map = {}
  box_to_keypoints_map = collections.defaultdict(list)
  box_to_track_ids_map = {}
  if not max_boxes_to_draw:
    max_boxes_to_draw = boxes.shape[0]
    
  for i in range(min(max_boxes_to_draw, boxes.shape[0])): # 0 =< i < 20
    #print("박스타입" + str(type(boxes.shape[0])) + str(boxes.shape[0]) + "i갑" + str(i) )
    
    if scores is None or scores[i] > min_score_thresh:
      #최소 값 이상 0부터 내가설정한 값까지
      box = tuple(boxes[i].tolist())
      if instance_masks is not None:
        box_to_instance_masks_map[box] = instance_masks[i]
      if instance_boundaries is not None:
        box_to_instance_boundaries_map[box] = instance_boundaries[i]
      if keypoints is not None:
        box_to_keypoints_map[box].extend(keypoints[i])
      if track_ids is not None:
        box_to_track_ids_map[box] = track_ids[i]
      if scores is None: #색
        box_to_color_map[box] = groundtruth_box_visualization_color
        
      else: #리스트 풀어서 박스 영역에 확률 표시
        display_str = '' #화면에 표시될 문자열
        if not skip_labels:
          if not agnostic_mode:
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name) #그렇게 문자열 저장
            
        if not skip_scores: #화면에 표시될 확률값 계산 형변
          if not display_str:
            display_str = '{}%'.format(int(100*scores[i]))
          else:
            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
            
        if not skip_track_ids and track_ids is not None: #위치값
          if not display_str:
            display_str = 'ID {}'.format(track_ids[i])
          else:
            display_str = '{}: ID {}'.format(display_str, track_ids[i])
            
        box_to_display_str_map[box].append(display_str)

        if agnostic_mode:
          box_to_color_map[box] = 'DarkOrange'
        elif track_ids is not None:
          prime_multipler = _get_multiplier_for_color_randomness()
          box_to_color_map[box] = STANDARD_COLORS[
              (prime_multipler * track_ids[i]) % len(STANDARD_COLORS)]
        else:
          box_to_color_map[box] = STANDARD_COLORS[
              classes[i] % len(STANDARD_COLORS)]

  # Draw all boxes onto image.
  # 모든 이미지에 박스를 그리게 되는데...
  
  for box, color in box_to_color_map.items():
    ymin, xmin, ymax, xmax = box


    #print("box값:" + str(box) + "깔라값 : " + str(color) )



    #type(box) = tuple 가로세로 정보를 갖고있음
    
    '''
      instance_masks: a numpy array of shape [N, image_height, image_width] with
      values ranging between 0 and 1, can be None.
      instance_masks : 모양이 [N, image_height, image_width] 인 numpy 배열
      0과 1 사이의 값은 None 일 수 있습니다.
    '''
    if instance_masks is not None: #null값이 아니라면 이미지 배열에 저장
      
      draw_mask_on_image_array(
          image,
          box_to_instance_masks_map[box],
          color=color
      )
      '''
      instance_boundaries: a numpy array of shape [N, image_height, image_width]
      with values ranging between 0 and 1, can be None.
      instance_boundaries : numpy 배열 모양 [N, image_height, image_width]
      0과 1 사이의 값으로 None이 될 수 있습니다.
      
      '''
    if instance_boundaries is not None: #null값이 아니라면 저
      draw_mask_on_image_array(
          image,
          box_to_instance_boundaries_map[box],
          color='red',
          alpha=1.0
      )
      
    draw_bounding_box_on_image_array(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display_str_map[box],
        use_normalized_coordinates=use_normalized_coordinates)
     
    #keypoints: a numpy array of shape [N, num_keypoints, 2], can be None
        
    if keypoints is not None:
      draw_keypoints_on_image_array(
          image,
          box_to_keypoints_map[box],
          color=color,
          radius=line_thickness / 2,
          use_normalized_coordinates=use_normalized_coordinates)

  return image


def add_cdf_image_summary(values, name):
  """Adds a tf.summary.image for a CDF plot of the values.

  Normalizes `values` such that they sum to 1, plots the cumulative distribution
  function and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    name: name for the image summary.
  """
  def cdf_plot(values):
    """Numpy function to plot CDF."""
    normalized_values = values / np.sum(values)
    sorted_values = np.sort(normalized_values)
    cumulative_values = np.cumsum(sorted_values)
    fraction_of_examples = (np.arange(cumulative_values.size, dtype=np.float32)
                            / cumulative_values.size)
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    ax.plot(fraction_of_examples, cumulative_values)
    ax.set_ylabel('cumulative normalized values')
    ax.set_xlabel('fraction of examples')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(
        1, int(height), int(width), 3)
    return image
  cdf_plot = tf.py_func(cdf_plot, [values], tf.uint8)
  tf.summary.image(name, cdf_plot)


def add_hist_image_summary(values, bins, name):
  """Adds a tf.summary.image for a histogram plot of the values.

  Plots the histogram of values and creates a tf image summary.

  Args:
    values: a 1-D float32 tensor containing the values.
    bins: bin edges which will be directly passed to np.histogram.
    name: name for the image summary.
  """

  def hist_plot(values, bins):
    """Numpy function to plot hist."""
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot('111')
    y, x = np.histogram(values, bins=bins)
    ax.plot(x[:-1], y)
    ax.set_ylabel('count')
    ax.set_xlabel('value')
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image = np.fromstring(
        fig.canvas.tostring_rgb(), dtype='uint8').reshape(
            1, int(height), int(width), 3)
    return image
  hist_plot = tf.py_func(hist_plot, [values, bins], tf.uint8)
  tf.summary.image(name, hist_plot)


class EvalMetricOpsVisualization(object):
  """Abstract base class responsible for visualizations during evaluation.

  Currently, summary images are not run during evaluation. One way to produce
  evaluation images in Tensorboard is to provide tf.summary.image strings as
  `value_ops` in tf.estimator.EstimatorSpec's `eval_metric_ops`. This class is
  responsible for accruing images (with overlaid detections and groundtruth)
  and returning a dictionary that can be passed to `eval_metric_ops`.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='evaluation_image'):
    """Creates an EvalMetricOpsVisualization.

    Args:
      category_index: A category index (dictionary) produced from a labelmap.
      max_examples_to_draw: The maximum number of example summaries to produce.
      max_boxes_to_draw: The maximum number of boxes to draw for detections.
      min_score_thresh: The minimum score threshold for showing detections.
      use_normalized_coordinates: Whether to assume boxes and kepoints are in
        normalized coordinates (as opposed to absolute coordiantes).
        Default is True.
      summary_name_prefix: A string prefix for each image summary.
    """

    self._category_index = category_index
    self._max_examples_to_draw = max_examples_to_draw
    self._max_boxes_to_draw = max_boxes_to_draw
    self._min_score_thresh = min_score_thresh
    self._use_normalized_coordinates = use_normalized_coordinates
    self._summary_name_prefix = summary_name_prefix
    self._images = []

  def clear(self):
    self._images = []

  def add_images(self, images):
    """Store a list of images, each with shape [1, H, W, C]."""
    if len(self._images) >= self._max_examples_to_draw:
      return

    # Store images and clip list if necessary.
    self._images.extend(images)
    if len(self._images) > self._max_examples_to_draw:
      self._images[self._max_examples_to_draw:] = []

  def get_estimator_eval_metric_ops(self, eval_dict):
    """Returns metric ops for use in tf.estimator.EstimatorSpec.

    Args:
      eval_dict: A dictionary that holds an image, groundtruth, and detections
        for a batched example. Note that, we use only the first example for
        visualization. See eval_util.result_dict_for_batched_example() for a
        convenient method for constructing such a dictionary. The dictionary
        contains
        fields.InputDataFields.original_image: [batch_size, H, W, 3] image.
        fields.InputDataFields.original_image_spatial_shape: [batch_size, 2]
          tensor containing the size of the original image.
        fields.InputDataFields.true_image_shape: [batch_size, 3]
          tensor containing the spatial size of the upadded original image.
        fields.InputDataFields.groundtruth_boxes - [batch_size, num_boxes, 4]
          float32 tensor with groundtruth boxes in range [0.0, 1.0].
        fields.InputDataFields.groundtruth_classes - [batch_size, num_boxes]
          int64 tensor with 1-indexed groundtruth classes.
        fields.InputDataFields.groundtruth_instance_masks - (optional)
          [batch_size, num_boxes, H, W] int64 tensor with instance masks.
        fields.DetectionResultFields.detection_boxes - [batch_size,
          max_num_boxes, 4] float32 tensor with detection boxes in range [0.0,
          1.0].
        fields.DetectionResultFields.detection_classes - [batch_size,
          max_num_boxes] int64 tensor with 1-indexed detection classes.
        fields.DetectionResultFields.detection_scores - [batch_size,
          max_num_boxes] float32 tensor with detection scores.
        fields.DetectionResultFields.detection_masks - (optional) [batch_size,
          max_num_boxes, H, W] float32 tensor of binarized masks.
        fields.DetectionResultFields.detection_keypoints - (optional)
          [batch_size, max_num_boxes, num_keypoints, 2] float32 tensor with
          keypoints.

    Returns:
      A dictionary of image summary names to tuple of (value_op, update_op). The
      `update_op` is the same for all items in the dictionary, and is
      responsible for saving a single side-by-side image with detections and
      groundtruth. Each `value_op` holds the tf.summary.image string for a given
      image.
    """
    if self._max_examples_to_draw == 0:
      return {}
    images = self.images_from_evaluation_dict(eval_dict)

    def get_images():
      """Returns a list of images, padded to self._max_images_to_draw."""
      images = self._images
      while len(images) < self._max_examples_to_draw:
        images.append(np.array(0, dtype=np.uint8))
      self.clear()
      return images

    def image_summary_or_default_string(summary_name, image):
      """Returns image summaries for non-padded elements."""
      return tf.cond(
          tf.equal(tf.size(tf.shape(image)), 4),
          lambda: tf.summary.image(summary_name, image),
          lambda: tf.constant(''))

    update_op = tf.py_func(self.add_images, [[images[0]]], [])
    image_tensors = tf.py_func(
        get_images, [], [tf.uint8] * self._max_examples_to_draw)
    eval_metric_ops = {}
    for i, image in enumerate(image_tensors):
      summary_name = self._summary_name_prefix + '/' + str(i)
      value_op = image_summary_or_default_string(summary_name, image)
      eval_metric_ops[summary_name] = (value_op, update_op)
    return eval_metric_ops

  @abc.abstractmethod
  def images_from_evaluation_dict(self, eval_dict):
    """Converts evaluation dictionary into a list of image tensors.

    To be overridden by implementations.

    Args:
      eval_dict: A dictionary with all the necessary information for producing
        visualizations.

    Returns:
      A list of [1, H, W, C] uint8 tensors.
    """
    raise NotImplementedError


class VisualizeSingleFrameDetections(EvalMetricOpsVisualization):
  """Class responsible for single-frame object detection visualizations."""

  def __init__(self,
               category_index,
               max_examples_to_draw=5,
               max_boxes_to_draw=20,
               min_score_thresh=0.2,
               use_normalized_coordinates=True,
               summary_name_prefix='Detections_Left_Groundtruth_Right'):
    super(VisualizeSingleFrameDetections, self).__init__(
        category_index=category_index,
        max_examples_to_draw=max_examples_to_draw,
        max_boxes_to_draw=max_boxes_to_draw,
        min_score_thresh=min_score_thresh,
        use_normalized_coordinates=use_normalized_coordinates,
        summary_name_prefix=summary_name_prefix)

  def images_from_evaluation_dict(self, eval_dict):
    return draw_side_by_side_evaluation_image(
        eval_dict, self._category_index, self._max_boxes_to_draw,
        self._min_score_thresh, self._use_normalized_coordinates)
