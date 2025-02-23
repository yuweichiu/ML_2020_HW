# -*- coding: utf-8 -*-
"""
Collect built models on Keras.
Created on : 2019/9/27
@author: Ivan Chiu
"""

import src.keras_tools as kst
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPool2D
from keras.layers import Flatten, LeakyReLU, UpSampling2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.initializers import Constant, truncated_normal
from keras.optimizers import SGD, Adam
from keras import backend as K


def k_autoencoder4(latent_dim, folder):
    inputs = kst.md_input((32, 32, 3))
    net = kst.conv2d(inputs, 64, (3, 3))  # 32,32,64
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 64, (3, 3))  # 32,32,64
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net)  # 16,16,64
    net = kst.conv2d(net, 128, (3, 3))  # 16,16,128
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 128, (3, 3))  # 16,16,128
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net)  # 8,8,128
    net = kst.conv2d(net, 256, (3, 3))  # 8,8,256
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 256, (3, 3))
    net = kst.flatten(net)  # 8*8*256
    net = kst.dense(net, 2048)  # 1024
    net = kst.activation(net, "relu")
    net = kst.dense(net, latent_dim)
    encode = kst.activation(net, "relu")
    net = kst.dense(encode, 2048) #19
    net = kst.activation(net, "relu")
    net = kst.dense(net, 8 * 8 * 256)
    net = kst.activation(net, "relu")
    net = kst.reshape(net, (8, 8, 256))  # 4,4,512
    # net = kst.deconv2d(net, 512, (3, 3))
    # net = kst.activation(net, "relu")
    # net = kst.deconv2d(net, 512, (3, 3))
    # net = kst.upsampling(net, 2)  # 8,8,512
    net = kst.deconv2d(net, 256, (3, 3))  # 8,8,256
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 256, (3, 3))
    net = kst.upsampling(net, 2)  # 16,16,256
    net = kst.deconv2d(net, 128, (3, 3))  # 16,16,128
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 128, (3, 3))
    net = kst.activation(net, "relu")
    net = kst.upsampling(net, 2)  # 32,32,128
    net = kst.deconv2d(net, 64, (3, 3))  # 32,32,64
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 64, (3, 3))
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 3, (3, 3))  # 32,32,3
    decode = kst.activation(net, "sigmoid")
    auto_encoder = Model(inputs, decode)
    net_struct = kst.model_summary(auto_encoder, print_out=True, save_dir=folder + "/model_summary.txt")

    encoder = Model(inputs, encode)
    encoded_input = kst.md_input(shape=(latent_dim,))
    decoding = auto_encoder.layers[19](encoded_input)
    for layer in auto_encoder.layers[20:]:
        decoding = layer(decoding)
    decoder = Model(encoded_input, decoding)

    return auto_encoder, encoder, decoder


def k_autoencoder3(latent_dim, folder):
    inputs = kst.md_input((32, 32, 3))
    net = kst.conv2d(inputs, 64, (3, 3))  # 32,32,64
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 64, (3, 3))  # 32,32,64
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net)  # 16,16,64
    net = kst.conv2d(net, 128, (3, 3))  # 16,16,128
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 128, (3, 3))  # 16,16,128
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net)  # 8,8,128
    net = kst.conv2d(net, 256, (3, 3))  # 8,8,256
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 256, (3, 3))
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net)  # 4,4,256
    net = kst.conv2d(net, 512, (3, 3))  # 4,4,512
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 512, (3, 3))
    net = kst.flatten(net)  # 4*4*512
    net = kst.dense(net, 2048)  # 1024
    net = kst.activation(net, "relu")
    net = kst.dense(net, latent_dim)
    encode = kst.activation(net, "relu")  # 23
    net = kst.dense(encode, 2048)
    net = kst.activation(net, "relu")
    net = kst.dense(net, 4 * 4 * 512)
    net = kst.activation(net, "relu")
    net = kst.reshape(net, (4, 4, 512))  # 4,4,512
    net = kst.deconv2d(net, 512, (3, 3))
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 512, (3, 3))
    net = kst.upsampling(net, 2)  # 8,8,512
    net = kst.deconv2d(net, 256, (3, 3))  # 8,8,256
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 256, (3, 3))
    net = kst.upsampling(net, 2)  # 16,16,256
    net = kst.deconv2d(net, 128, (3, 3))  # 16,16,128
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 128, (3, 3))
    net = kst.activation(net, "relu")
    net = kst.upsampling(net, 2)  # 32,32,128
    net = kst.deconv2d(net, 64, (3, 3))  # 32,32,64
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 64, (3, 3))
    net = kst.activation(net, "relu")
    net = kst.deconv2d(net, 3, (3, 3))  # 32,32,3
    decode = kst.activation(net, "sigmoid")
    auto_encoder = Model(inputs, decode)
    net_struct = kst.model_summary(auto_encoder, print_out=True, save_dir=folder + "/model_summary.txt")

    encoder = Model(inputs, encode)
    encoded_input = kst.md_input(shape=(latent_dim,))
    decoding = auto_encoder.layers[24](encoded_input)
    for layer in auto_encoder.layers[25:]:
        decoding = layer(decoding)
    decoder = Model(encoded_input, decoding)

    return auto_encoder, encoder, decoder


def k_inception_v3(n_class, dp):
    # TODO: make sure the reliability"
    inputs = kst.md_input((299, 299, 1))
    # inputs = K.resize_images(inputs, 299, 299, data_format="channels_last")
    net = kst.conv2d(inputs, 32, (3, 3), stride=2, padding="VALID")
    net = kst.batch_norm(net)
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 32, (3, 3), padding="VALID")
    net = kst.batch_norm(net)
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 64, (3, 3), padding="SAME")
    net = kst.batch_norm(net)
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net, kernel_size=(3, 3), stride=2, padding="VALID")
    net = kst.conv2d(net, 80, (1, 1), padding="VALID")
    net = kst.batch_norm(net)
    net = kst.activation(net, "relu")
    net = kst.conv2d(net, 192, (3, 3), padding="VALID")
    net = kst.batch_norm(net)
    net = kst.activation(net, "relu")
    net = kst.maxpool2d(net, (3, 3), padding="VALID")

    # mixed 5b:
    branch0 = kst.conv2d(net, 64, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 48, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 64, (5, 5))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 64, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 96, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 96, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 32, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 5c:
    branch0 = kst.conv2d(net, 64, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 48, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 64, (5, 5))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 64, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 96, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 96, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 64, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 5d:
    branch0 = kst.conv2d(net, 64, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 48, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 64, (5, 5))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 64, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 96, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 96, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 64, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 6a:
    branch0 = kst.conv2d(net, 348, (3, 3), stride=2, padding="VALID")
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 64, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 96, (3, 3))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 96, (3, 3), stride=2, padding="VALID")
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.maxpool2d(net, (3, 3), padding="VALID")
    net = kst.concat([branch0, branch1, branch2], axis=3)

    # mixed 6b:
    branch0 = kst.conv2d(net, 192, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 128, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 128, (1, 7))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (7, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 128, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 128, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 128, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 128, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 192, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 6c:
    branch0 = kst.conv2d(net, 192, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 160, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 160, (1, 7))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (7, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 160, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 160, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 160, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 160, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 192, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 6d:
    branch0 = kst.conv2d(net, 192, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 160, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 160, (1, 7))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (7, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 160, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 160, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 160, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 160, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 192, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 6e:
    branch0 = kst.conv2d(net, 192, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 192, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (1, 7))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (7, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.conv2d(net, 192, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (7, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch2 = kst.conv2d(branch2, 192, (1, 7))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, 'relu')
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 192, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)
    aux_logits = net

    # mixed 7a:
    branch0 = kst.conv2d(net, 192, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch0 = kst.conv2d(branch0, 320, (3, 3), stride=2, padding="VALID")
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 192, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (1, 7))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (7, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1 = kst.conv2d(branch1, 192, (3, 3), stride=2, padding="VALID")
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch2 = kst.maxpool2d(net, (3, 3), padding="VALID")
    net = kst.concat([branch0, branch1, branch2], axis=3)

    # mixed 7b:
    branch0 = kst.conv2d(net, 320, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 384, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1a = kst.conv2d(branch1, 384, (1, 3))
    branch1a = kst.batch_norm(branch1a)
    branch1a = kst.activation(branch1a, "relu")
    branch1b = kst.conv2d(branch1, 384, (3, 1))
    branch1b = kst.batch_norm(branch1b)
    branch1b = kst.activation(branch1b, "relu")
    branch1 = kst.concat([branch1a, branch1b], axis=3)
    branch2 = kst.conv2d(net, 448, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, "relu")
    branch2 = kst.conv2d(branch2, 384, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, "relu")
    branch2a = kst.conv2d(branch2, 384, (1, 3))
    branch2a = kst.batch_norm(branch2a)
    branch2a = kst.activation(branch2a, "relu")
    branch2b = kst.conv2d(branch1, 384, (3, 1))
    branch2b = kst.batch_norm(branch2b)
    branch2b = kst.activation(branch2b, "relu")
    branch2 = kst.concat([branch2a, branch2b], axis=3)
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 192, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # mixed 7c:
    branch0 = kst.conv2d(net, 320, (1, 1))
    branch0 = kst.batch_norm(branch0)
    branch0 = kst.activation(branch0, "relu")
    branch1 = kst.conv2d(net, 384, (1, 1))
    branch1 = kst.batch_norm(branch1)
    branch1 = kst.activation(branch1, "relu")
    branch1a = kst.conv2d(branch1, 384, (1, 3))
    branch1a = kst.batch_norm(branch1a)
    branch1a = kst.activation(branch1a, "relu")
    branch1b = kst.conv2d(branch1, 384, (3, 1))
    branch1b = kst.batch_norm(branch1b)
    branch1b = kst.activation(branch1b, "relu")
    branch1 = kst.concat([branch1a, branch1b], axis=3)
    branch2 = kst.conv2d(net, 448, (1, 1))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, "relu")
    branch2 = kst.conv2d(branch2, 384, (3, 3))
    branch2 = kst.batch_norm(branch2)
    branch2 = kst.activation(branch2, "relu")
    branch2a = kst.conv2d(branch2, 384, (1, 3))
    branch2a = kst.batch_norm(branch2a)
    branch2a = kst.activation(branch2a, "relu")
    branch2b = kst.conv2d(branch1, 384, (3, 1))
    branch2b = kst.batch_norm(branch2b)
    branch2b = kst.activation(branch2b, "relu")
    branch2 = kst.concat([branch2a, branch2b], axis=3)
    branch3 = kst.avgpool2d(net, (3, 3))
    branch3 = kst.conv2d(branch3, 192, (1, 1))
    branch3 = kst.batch_norm(branch3)
    branch3 = kst.activation(branch3, "relu")
    net = kst.concat([branch0, branch1, branch2, branch3], axis=3)

    # auxlogits:
    aux_logits = kst.avgpool2d(aux_logits, (5, 5), stride=3, padding="VALID")
    aux_logits = kst.conv2d(aux_logits, 128, (1, 1))
    aux_logits = kst.batch_norm(aux_logits)
    aux_logits = kst.activation(aux_logits, "relu")
    aux_logits = kst.conv2d(aux_logits, 768, (5, 5), padding="VALID")
    aux_logits = kst.batch_norm(aux_logits)
    aux_logits = kst.activation(aux_logits, "relu")
    aux_logits = kst.conv2d(aux_logits, n_class, (1, 1))
    aux_logits = kst.flatten(aux_logits)
    # aux_logits = kst.squeeze(aux_logits, 1)
    # aux_logits = kst.squeeze(aux_logits, 1)
    aux_predict = kst.activation(aux_logits, "softmax")

    # Logits:
    net = kst.avgpool2d(net, (8, 8), padding="VALID")
    net = kst.dropout(net, dp)
    logits = kst.conv2d(net, n_class, (1, 1))
    # logits = kst.squeeze(logits, 1)
    # logits = kst.squeeze(logits, 1)
    logits = kst.flatten(logits)
    predict = kst.activation(logits, "softmax")
    inception_v3 = Model(inputs, [predict, aux_predict])
    net_struct = kst.model_summary(inception_v3, print_out=True)
    # aux_inception = Model(inputs, aux_predict)

    return inception_v3


