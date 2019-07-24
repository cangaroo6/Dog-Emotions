from math import degrees
import math

import operator
from functools import reduce
import numpy as np
import pandas as pd
import face_recognition
import warnings
import matplotlib.pyplot as plt

from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist

import cv2

class No_Preprocessing:

    def __init__(self, img_width, img_height):
        self.img_width = img_width
        self.img_height = img_height

    def extract_and_prepare_pixels(self, pixels):
        """
        Takes in a string (pixels) that has space separated integer values and returns an array which includes the
        pixels for all images.
        """
        pixels_as_list = [item[0] for item in pixels.values.tolist()]
        np_image_array = []
        for index, item in enumerate(pixels_as_list):
            # split space separated ints
            pixel_data = item.split()
            img_size_row = img_size_col = 256
            if len(pixel_data) % 490 == 0:
                img_size_row = 490
                img_size_col = 640
            elif len(pixel_data) == 10000:
                img_size_row = img_size_col = 100

            data = np.zeros((img_size_row, img_size_col), dtype=np.uint8)

            # Loop through rows
            for i in range(0, img_size_row):
                # (0 = 0), (1 = 47), (2 = 94), ...
                pixel_index = i * img_size_col
                # (0 = [0:47]), (1 = [47: 94]), (2 = [94, 141]), ...
                data[i] = pixel_data[pixel_index:pixel_index + img_size_col]

            np_image_array.append(np.array(data))
        np_image_array = np.array(np_image_array)
        return np_image_array


    def predict_emotion(self, model, img):
        """
        Use a trained model to predict emotional state
        """

        emotion = 'None'

        prediction = model.predict(img)
        prediction_ = np.argmax(prediction)

        if prediction_ == 0:
            emotion = 'Angry'
        elif prediction_ == 1:
            emotion = 'Scared'
        elif prediction_ == 2:
            emotion = 'Happy'
        elif prediction_ == 3:
            emotion = 'Sad'

        d = {'emotion': ['Angry', 'Scared', 'Happy', 'Sad'], 'prob': prediction[0]}
        df = pd.DataFrame(d, columns=['emotion', 'prob'])

        return df

