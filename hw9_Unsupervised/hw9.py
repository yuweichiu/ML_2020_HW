# -*- coding: utf-8 -*-
"""
Created on 2020/6/25 下午 04:45
@author: Ivan Y.W.Chiu

Data Source:
# https://drive.google.com/uc?id=1BZb2AqOHHaad7Mo82St1qTBaXo_xtcUc  # trainX.npy
# https://drive.google.com/uc?id=152NKCpj8S_zuIx3bQy0NN5oqpvBjdPIq  # valX.npy
# https://drive.google.com/uc?id=1_hRGsFtm5KEazUg2ZvPZcuNScGF-ANh4  # valY.npy
"""

import sys, os, time
import numpy as np
import pandas as pd
import cv2
import src.keras_models as kmd
import src.keras_tools as knt
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from MulticoreTSNE import MulticoreTSNE as TSNE



# In[]
# Data Processing
def data_preprocessing(x):
    x = x.astype(dtype=np.float32)
    x = x / 255
    return x

trainX = np.load("./data/hw9_data/trainX_new.npy")
valX = np.load("./data/hw9_data/valX.npy")
valY = np.load("./data/hw9_data/valY.npy")
trainX = data_preprocessing(trainX)
valX = data_preprocessing(valX)


# In[]
# Model definition
epoch = 100
latent_dim = 256

# In[]
# Training model
K.clear_session()
tn = time.localtime()
project = "./logs/hw9/D{0:4d}{1:02d}{2:02d}T{3:02d}{4:02d}".format(tn[0], tn[1], tn[2], tn[3], tn[4])
os.mkdir(project)
auto_encoder, encoder, decoder = kmd.k_autoencoder4(latent_dim, project)
auto_encoder.compile(loss="MSE", optimizer=Adam(lr=0.0001))
mc = ModelCheckpoint(os.path.join(project, 'best_model.h5'), monitor='val_loss', mode='min',
                     verbose=1, save_best_only=True)
hist = auto_encoder.fit(trainX, trainX,
                        batch_size=256,
                        epochs=epoch,
                        callbacks=[mc],
                        validation_data=(valX, valX)) # validation_split=0.2

loss_train = hist.history['loss']
loss_valid = hist.history['val_loss']

# In[]
# Load trained model
project = "logs\hw9\D20200625T1712"
latent_dim = 256
auto_encoder = load_model(project + "/best_model.h5")
# inputs = knt.md_input((32, 32, 3))
inputs = auto_encoder.input
encoding = auto_encoder.layers[1](inputs)
for layer in auto_encoder.layers[2:24]:
    encoding = layer(encoding)
encoder = Model(inputs, encoding)

encoded_input = knt.md_input(shape=(latent_dim,))
decoding = auto_encoder.layers[24](encoded_input)
for layer in auto_encoder.layers[25:]:
    decoding = layer(decoding)
decoder = Model(encoded_input, decoding)


# In[]
# Training history:
fig2 = plt.figure()
plt.rcParams.update({'font.size': 14})
ax3 = fig2.add_subplot(1, 1, 1)
ax3.plot(range(epoch), loss_train, label="Training")
ax3.plot(range(epoch), loss_valid, label="Validation")
ax3.legend()
ax3.set_xlabel("epoch")
ax3.set_ylabel("MSE")
ax3.set_title("Minimum validated MSE: " + "{0:.5f}".format(min(loss_valid)))
plt.tight_layout()
plt.show(block=False)
fig2.savefig(project + "/training_curve.png", dpi=300)
# plt.close("all")

# In[]
# View some image in training data:
n2see = 10
x_test = trainX[-n2see:]
encoded = encoder.predict(x_test)
y_pred = decoder.predict(encoded)

fig = plt.figure("AutoEncoder", (20, 8))
for f in range(n2see):
    ax1 = fig.add_subplot(2, n2see, f + 1)
    ax1.imshow(x_test[f])
    ax1.set_axis_off()
    ax2 = fig.add_subplot(2, n2see, f + 1 + n2see)
    ax2.imshow(y_pred[f])
    ax2.set_axis_off()

# In[]
def predict(latents):
    # Dimension reduction 1st:
    pca = PCA(n_components=120, copy=False, whiten=True, svd_solver="full")
    latents = pca.fit_transform(latents)

    # Dimension reduction 2nd:
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=100000)
    latents = tsne.fit_transform(latents)
    # tsne = TSNE(n_jobs=6, n_components=2, verbose=1, perplexity=50, n_iter=5000)
    # latents = tsne.fit_transform(latents)
    # np.savetxt(project + "/encode.txt", tsne_code)

    # Clustering:
    kmeans = KMeans(n_clusters=2, random_state=0).fit(latents)
    pred = kmeans.labels_

    return pred, latents

def cal_acc(ans, pred):
    correct = np.sum(ans == pred)
    acc = correct / ans.shape[0]
    return max(acc, 1-acc)


def visualize_scatter(feat, label, savefig=None):
    x = feat[:, 0]
    y = feat[:, 1]
    id0 = np.squeeze(np.argwhere(label == 0))
    id1 = np.squeeze(np.argwhere(label == 1))
    fig = plt.figure()
    plt.rcParams.update({'font.size': 14})
    ax = fig.add_subplot(1, 1, 1)
    scatter1 = ax.scatter(x[id0], y[id0], c="b", label="Label 0")
    scatter2 = ax.scatter(x[id1], y[id1], c="r", label="Label 1")
    ax.legend(loc='best')
    ax.set_xlabel("Feature #1")
    ax.set_ylabel("Feature #2")
    ax.set_title("Clustering")
    plt.tight_layout()
    plt.show(block=False)
    if savefig:
        fig.savefig(savefig, dpi=300)
    return fig, ax

# In[]
latents = encoder.predict(trainX, batch_size=256)
pred, train_embed = predict(latents)

results_dict = {'id': list(range(pred.shape[0])), 'label': pred}
results_df = pd.DataFrame(results_dict)
results_df["label"] = np.abs(1 - results_df["label"].values)

visualize_scatter(train_embed, pred)

results_df.to_csv(project + "/prediction.csv", index=False)

# In[]
# Check validation
latent_val = encoder.predict(valX, batch_size=256)
pred_val, val_embed = predict(latent_val)
results_val = {'id': list(range(pred_val.shape[0])), 'label': pred_val}
results_val_df = pd.DataFrame(results_val)

acc = cal_acc(valY, pred_val)
visualize_scatter(val_embed, pred_val, project + "/clustering.png")
