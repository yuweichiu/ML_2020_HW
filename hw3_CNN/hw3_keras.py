# -*- coding: utf-8 -*-
"""
Created on 2020/7/19
@author: Ivan Y.W.Chiu

Data Source:
# https://drive.google.com/uc?id=19CzXudqN58R3D-1G8KeFWk8UDQwlb8is  # trainX.npy
"""

from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras import backend as K
import src.keras_tools as kst
import src.config as Config
import src.utils as utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, cv2

# In[]
class FoodConfig(Config.ProjectConfig):
    NAME = "Food"  # your project name
    DATA_PATH = os.path.join(".//data//hw3_data//food-11")  # the folder path you want to load
    CLASS_NUM = 11  # the number of class in the dataset used in this project
    CLASS_NAME_LIST = []  # the list of the class names
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    IMAGE_CHANNEL = 3


def read_image_and_class(which, config):
    image_list = os.listdir(os.path.join(config.DATA_PATH, which))
    x = np.zeros((len(image_list), config.IMAGE_WIDTH, config.IMAGE_HEIGHT, config.IMAGE_CHANNEL), dtype=np.uint8)
    if which == "testing":
        y = None
    else:
        y = np.zeros((len(image_list)))
    for xid, xname in enumerate(image_list):
        img = cv2.imread(os.path.join(config.DATA_PATH, which, xname))
        x[xid, :, :] = cv2.resize(img, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        if which != "testing":
            y[xid] = image_list[xid].split("_")[0]
    return x, y

pj_config = FoodConfig()

# In[]
x_train, y_train = read_image_and_class("training", pj_config)
x_valid, y_valid = read_image_and_class("validation", pj_config)
x_test, _ = read_image_and_class("testing", pj_config)

# In[]
if pj_config.MANUAL_GPU_MEMORY_GROWTH is True:
    sess = utils.tf_session_setting()

