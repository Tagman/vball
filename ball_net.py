import sys
import numpy as np
import cv2 as cv
import os

from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array

size = 32
dim = 3


def check_pic(pic):
    img = cv.resize(pic, (size, size))
    # cv.imshow('detect', img)
    # cv.waitKey(0)
    # cv.destroyWindow('detect')

    img = np.reshape(img, [1, size, size, dim])
    prediction = loaded_model.predict(img)
    return prediction


# json_file = open('ball-net/model/model_austria.json', 'r')
json_file = open('ball-net/model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("ball-net/model/model.h5")
# loaded_model.load_weights("ball-net/model/model_austria.h5")
