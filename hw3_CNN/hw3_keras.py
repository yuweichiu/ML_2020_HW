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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time




