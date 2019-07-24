import cv2
from keras.models import load_model
from helper import No_Preprocessing
import dlib
from imutils import face_utils
import imutils
import numpy as np
import math

# ---------------------------------------------------------

# image size for prediction
img_width = 100
img_height = 100
# scale factor for preprocessing
picSize = 200
rotation = True

# face detector
pathDet = '../faceDetectors/dogHeadDetector.dat'
detector = dlib.cnn_face_detection_model_v1(pathDet)

# landmarks detector
pathPred = '../faceDetectors/landmarkDetector.dat'
predictor = dlib.shape_predictor(pathPred)

# helper class
helper = No_Preprocessing(img_width, img_height)

# load model
#model = load_model('models/classifierRotatedOn100Ratio90.h5')
model = load_model('../models/classifierRotatedOn100Ratio90Epochs100.h5')

# ---------------------------------------------------------
#returns the first dog face found
def preprocess(orig):
  imageList = []  # save image for each face
  faces = []

  if orig is not None:
    # resize
    height, width, channels = orig.shape  # read size
    ratio = picSize / height
    image = cv2.resize(orig, None, fx=ratio, fy=ratio)

    # color gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect face(s)
    dets = detector(gray, upsample_num_times=1)

    for i, d in enumerate(dets):
      # save coordinates
      x1 = max(int(d.rect.left() / ratio), 1)
      y1 = max(int(d.rect.top() / ratio), 1)
      x2 = min(int(d.rect.right() / ratio), width - 1)
      y2 = min(int(d.rect.bottom() / ratio), height - 1)

      # detect landmarks
      shape = face_utils.shape_to_np(predictor(gray, d.rect))
      points = []
      index = 0
      for (x, y) in shape:
        x = int(round(x / ratio))
        y = int(round(y / ratio))
        index = index + 1
        if index == 3 or index == 4 or index == 6:
          points.append([x, y])
      points = np.array(points) # right eye, nose, left eye

      # rotate
      if rotation == True:
        xLine = points[0][0] - points[2][0]
        if points[2][1] < points[0][1]:
          yLine = points[0][1] - points[2][1]
          angle = math.degrees(math.atan(yLine / xLine))
        else:
          yLine = points[2][1] - points[0][1]
          angle = 360 - math.degrees(math.atan(yLine / xLine))
        rotated = imutils.rotate(orig, angle)
        # detectFace(rotated, picSize)

      cv2.polylines(orig, [points], True, (0, 255, 0), 1)  # draw polygon
      cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 1) #draw rectangle
      imageList.append(orig)

      little = cv2.resize((rotated[y1:y2, x1:x2]), (img_width, img_height))  # crop and resize

      #pixel = skimage.io.imread(pathResult)
      pixel = cv2.cvtColor(little, cv2.COLOR_BGR2GRAY)
      x = np.expand_dims(pixel, axis=0)
      x = x.reshape((-1, 100, 100, 1))
      imageList.append(x)
      faces = [x1, y1]
  return imageList, faces  # order: marked, x, face

# ---------------------------------------------------------

def analyze_facial_emotions(image):
  images, face = preprocess(image)

  if images is not None and images != []:  # found face on image
    df = helper.predict_emotion(model, images[1])

    # sort and extract most probable emotion
    df = df.sort_values(by='prob', ascending=False)
    emotion = df['emotion'].values[0]
    prob = str(round((df['prob'].values[0]) * 100, 2))

    return emotion, face
  return '', [50,50]

# ---------------------------------------------------------

def face_recognition():
    # 3. Start WebCam & Analyse Faces
    cap = cv2.VideoCapture(0)

    process_this_frame = True
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, image = cap.read()
        if process_this_frame:
            emotion, face_location = analyze_facial_emotions(image)

        if emotion != '':
          print(emotion)
          process_this_frame = not process_this_frame

          cv2.putText(image, emotion, (int(face_location[0]), int(face_location[1])), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('img', image)

        if emotion != '':
          k = cv2.waitKey(100) & 0xff
        else:
        #time.sleep(1)
          k = cv2.waitKey(1) & 0xff

        # press 'e' to exit the webcam application
        if k == 101:
          cv2.destroyAllWindows()
          cap.release()
          break

face_recognition()