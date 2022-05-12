import streamlit as st
from PIL import Image
import numpy as np
import io
import cv2


st.title('目を拡大するよ')

upload_file = st.file_uploader('画像を選んでください', type='jpeg')

eye_resize = st.slider('目を拡大してね', 1, 50, 0)
resize1 = (1+eye_resize/100)
'拡大率：', resize1

if upload_file is not None:
  img = Image.open(upload_file)
  img = np.array(img)
  src = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  src_g  = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  #バイナリーデータを取得
  # with io.BytesIO() as output:
  #   img.save(output, format='JPEG')
  #   binary_img = output.getvalue()
  face_cascade_path = "../opt/anaconda3/envs/python=3.5/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml"
  eye_cascade_path = "../opt/anaconda3/envs/python=3.5/lib/python3.9/site-packages/cv2/data/haarcascade_eye.xml"
  left_eye_cascade_path = "../opt/anaconda3/envs/python=3.5/lib/python3.9/site-packages/cv2/data/haarcascade_lefteye_2splits.xml"
  right_eye_cascade_path = "../opt/anaconda3/envs/python=3.5/lib/python3.9/site-packages/cv2/data/haarcascade_righteye_2splits.xml"

  face_cascade = cv2.CascadeClassifier(face_cascade_path)
  eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
  left_eye_cascade = cv2.CascadeClassifier(left_eye_cascade_path)
  right_eye_cascade = cv2.CascadeClassifier(right_eye_cascade_path)

  

  faces = face_cascade.detectMultiScale(src_g);
  resize = resize1;
  blur_num = 3;

  for (x, y, w, h) in faces:
    cv2.rectangle(src, (x,y), (x + w, y + h), (255, 0, 255), 2)
    face = src[y: y+h, x: x+w]
    face_g = src_g[y: y+h, x: x+w]
    left_eyes = left_eye_cascade.detectMultiScale(face_g)
    right_eyes = right_eye_cascade.detectMultiScale(face_g)
    
    for (ex, ey, ew, eh) in left_eyes:
      left_eye = src[y+ey : y+ey+eh, x+ex : x+ex+ew]
      # cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
      trim_leye = cv2.resize(left_eye, dsize = None, fx=resize, fy=resize)
      lheight, lwidth = trim_leye.shape[:2]
      src[y+ey-int(eh*(resize-1)/2): y+ey+lheight-int(eh*(resize-1)/2), x+ex-int(ew*(resize-1)/2): x+ex+lwidth-int(ew*(resize-1)/2)] = trim_leye
      ltreheight, ltrewidth = trim_leye.shape[:2]

    for (rx, ry, rw, rh) in right_eyes:
      right_eye = src[y+ry : y+ry+rh, x+rx : x+rx+rw]
      trim_reye = cv2.resize(right_eye, dsize = None, fx=resize, fy=resize)
      rheight, rwidth = trim_reye.shape[:2]
      src[y+ry-int(rh*(resize-1)/2): y+ry+rheight-int(rh*(resize-1)/2), x+rx-int(rw*(resize-1)/2): x+rx+rwidth-int(rw*(resize-1)/2)] = trim_reye
      treheight, trewidth = trim_reye.shape[:2]
      # cv2.rectangle(face, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)

  def mosaic_area(image, x, y, width, height, blur_num):
    dst =image.copy()
    for i in range(blur_num):
      dst[y:y+height, x:x+width] = cv2.GaussianBlur(dst[y:y+height, x:x+width], (3,3),3)
    return dst

    if right_eye_cascade is not None:
      #右目拡大後左上移動
      reyex = x+rx-int(rw*(resize-1)/2)
      reyey = y+ry-int(rh*(resize-1)/2)

      ##右目
      #下
      src = mosaic_area(src, reyex-1, reyey+treheight-1, trewidth+2, 2, blur_num)
      #左
      src = mosaic_area(src, reyex-1, reyey-1, 2, treheight+2, blur_num)
      #上
      src = mosaic_area(src, reyex-1, reyey-1, trewidth+2, 2, blur_num)
      #右
      src = mosaic_area(src, reyex+trewidth-1, reyey-1, 2, treheight+2, blur_num)
    
    if left_eye_cascade is not None:
      leyex = x+ex-int(ew*(resize-1)/2)
      leyey = y+ey-int(eh*(resize-1)/2)
      #左目
      #下
      src = mosaic_area(src, leyex-1, leyey+ltreheight-1, ltrewidth+2, 2, blur_num)
      #左
      src = mosaic_area(src, leyex-1, leyey-1, 2, ltreheight+2, blur_num)
      #上
      src = mosaic_area(src, leyex-1, leyey-1, ltrewidth+2, 2, blur_num)
      #右
      src = mosaic_area(src, leyex+ltrewidth-1, leyey-1, 2, ltreheight+2, blur_num)

  src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
  st.image(src, caption='Uploaded Image.', use_column_width=True)









