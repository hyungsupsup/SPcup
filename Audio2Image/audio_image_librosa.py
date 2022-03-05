%matplotlib inline
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from collections import defaultdict, Counter
from scipy import signal
import numpy as np
import librosa
import random as rn
from keras.layers import Dense
from keras import Input
#from keras.engine import Model
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense, TimeDistributed, Dropout, Bidirectional, GRU, BatchNormalization, Activation, LeakyReLU, LSTM, Flatten, RepeatVector, Permute, Multiply, Conv2D, MaxPooling2D

from google.colab import drive

drive.mount('/content/drive')

data_dir = '/content/drive/MyDrive/Dataset/audiofile_shorted500/'

#sr : 오디오의 초당 샘플링 수, wav : 시계열 데이터
# 알고리즘 0
wav, sr = librosa.load(data_dir + '0_1.wav', sr = None)
print('sr:', sr)
print('wav shape:', wav.shape)
print('length:', wav.shape[0]/float(sr), 'secs')

#raw wave
print(plt.plot(wav))
#print(plt.plot(wav[0:500]))

import librosa.display

X = librosa.stft(wav)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()

# 알고리즘 0
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.title('data 0')

# 알고리즘 1
wav1, sr1 = librosa.load(data_dir + '1_1.wav', sr = None)
X1 = librosa.stft(wav1)
Xdb1 = librosa.amplitude_to_db(abs(X1))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb1, sr=sr1, x_axis='time', y_axis='log')
plt.title('data 1')

# 알고리즘 2
wav2, sr2 = librosa.load(data_dir + '2_1.wav', sr = None)
X2 = librosa.stft(wav2)
Xdb2 = librosa.amplitude_to_db(abs(X2))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb2, sr=sr2, x_axis='time', y_axis='log')
plt.title('data 2')

# 알고리즘 3
wav3, sr3 = librosa.load(data_dir + '3_1.wav', sr = None)
X3 = librosa.stft(wav3)
Xdb3 = librosa.amplitude_to_db(abs(X3))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb3, sr=sr3, x_axis='time', y_axis='log')
plt.title('data 3')

# 알고리즘 4
wav4, sr4 = librosa.load(data_dir + '4_1.wav', sr = None)
X4 = librosa.stft(wav4)
Xdb4 = librosa.amplitude_to_db(abs(X4))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb4, sr=sr4, x_axis='time', y_axis='log')
plt.title('data 4')

plt.colorbar()


# 알고리즘 1
wav1, sr1 = librosa.load(data_dir + '1_1.wav', sr = None)
X1 = librosa.stft(wav1)
Xdb1 = librosa.amplitude_to_db(abs(X1))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb1, sr=sr1, x_axis='time', y_axis='log')
plt.colorbar()

# 알고리즘 2
wav2, sr2 = librosa.load(data_dir + '2_1.wav', sr = None)
X2 = librosa.stft(wav2)
Xdb2 = librosa.amplitude_to_db(abs(X2))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb2, sr=sr2, x_axis='time', y_axis='log')
plt.colorbar()

# 알고리즘 3
wav3, sr3 = librosa.load(data_dir + '3_1.wav', sr = None)
X3 = librosa.stft(wav3)
Xdb3 = librosa.amplitude_to_db(abs(X3))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb3, sr=sr3, x_axis='time', y_axis='log')
plt.colorbar()

# 알고리즘 4
wav4, sr4 = librosa.load(data_dir + '4_1.wav', sr = None)
X4 = librosa.stft(wav4)
Xdb4 = librosa.amplitude_to_db(abs(X4))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb4, sr=sr4, x_axis='time', y_axis='log')
plt.colorbar()

import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
from PIL import Image
import pathlib
import csv 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
from keras import layers

### wav 파일 스펙토그램 png 파일로 변환 후 저장하기
# for filename in os.listdir(f'/content/drive/MyDrive/Dataset/audiofile_shorted500'):
#     audioname = f'/content/drive/MyDrive/Dataset/audiofile_shorted500/{filename}'
#     y, sr = librosa.load(audioname, mono=True, duration=5)
#     plt.specgram(y, NFFT=2048, Fs=16000, Fc=0, noverlap=128, sides='default', mode='default', scale='dB');
#     plt.axis('off');
#     plt.savefig(f'/content/drive/MyDrive/Dataset/audio_image_data/{filename[:-3].replace(".", "")}.png')
#     plt.clf()

import os, re, glob
import cv2

image_dir = '/content/drive/MyDrive/Dataset/audio_image_data/'
categories = ["0","1","2","3","4"]
num_classes = len(categories)
  
image_w = 432
image_h = 288

X = []
Y = []

for idex, category in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
  
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            print(type(img))
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/255)   # 255로 바꿔야 하나??? 
            Y.append(label)
 
X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)

X_train1 = np.concatenate((X[0:5000],X[6000:11000],X[12000:17000],X[18000:23000],X[24000:29000],X[30000:35000],X[36000:41000],X[42000:47000],X[48000:53000],X[54000:59000]), axis = 0)
X_test1 = np.concatenate((X[5000:6000],X[11000:12000],X[17000:1800],X[23000:24000],X[29000:30000],X[35000:36000],X[41000:42000],X[47000:48000],X[53000:54000],X[59000:60000]), axis=0)
Y_train1 = np.concatenate((Y[0:5000],Y[6000:11000],Y[12000:17000],Y[18000:23000],Y[24000:29000],Y[30000:35000],Y[36000:41000],Y[42000:47000],Y[48000:53000],Y[54000:59000]), axis = 0)
Y_test1 = np.concatenate((Y[5000:6000],Y[11000:12000],Y[17000:1800],Y[23000:24000],Y[29000:30000],Y[35000:36000],Y[41000:42000],Y[47000:48000],Y[53000:54000],Y[59000:60000]), axis=0)

print(X_train1.shape)
print(X_test1.shape)
print(Y_train1.shape)
print(Y_test1.shape)
xy = (X_train1, X_test1, Y_train1, Y_test1)

np.save("./img_data.npy", xy)
