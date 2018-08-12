import keras as k
import tensorflow as tf
import pandas as pd
from AceData import Ace
import os
import glob
from scipy.signal import stft
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, Activation
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras import optimizers


class RNN():
    def __init__(self,x,y):
        self.x =x
        self.label = y
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.label,test_size=0.0,random_state=1)


    def train(self):
        # expected input data shape: (batch_size, timesteps, data_dim)
        data_dim= 33 #number frequency component ion the signal
        timesteps = 129
        nb_class = 5
        #lets stack 3 layers of LSTM
        model = Sequential()
        model.add(LSTM(units=256, return_sequences= True, input_shape=(timesteps,data_dim)))#returns a sequence of vector dimension 100
        model.add(LSTM(units=128, return_sequences=True)) # returns a sequence of vector of dimension 100
        model.add(LSTM(units=64))
        model.add(Dense(nb_class, activation='softmax'))
        adam = optimizers.Adam(lr=0.00001)
        model.compile(loss='categorical_crossentropy',optimizer= adam,metrics=['accuracy'])
        model.summary()
        model.fit(self.x_train, self.y_train, epochs=100,validation_split= 0.1)




class FFT:
    def __init__(self, lamb_signal_obj):
        self.signal = lamb_signal_obj.sensor['s0']
        self.fs = lamb_signal_obj.setup['sampling_rate']

    @property
    def st_ft(self):
        f, t, Zxx = stft(self.signal.reshape((-1)),fs= self.fs, detrend ='linear')
        return f, t, Zxx

    def Short_time_fft(self,fft_size=256, overlap_fac=0.5):
        # data = a numpy array containing the signal to be processed
        # fs = a scalar which is the sampling frequency of the data
        fs = self.fs
        hop_size = np.int32(np.floor(fft_size * (1 - overlap_fac)))
        pad_end_size = fft_size  # the last segment can overlap the end of the data array by no more than one window size
        total_segments = np.int32(np.ceil(len(self.signal) / np.float32(hop_size)))
        t_max = len(self.signal) / np.float32(fs)

        window = np.hanning(fft_size)  # our half cosine window
        inner_pad = np.zeros(fft_size)  # the zeros which will be used to double each segment size

        proc = np.concatenate((self.signal.reshape((-1)), np.zeros(pad_end_size)))  # the data to process
        result = np.empty((total_segments, fft_size), dtype=np.float32)  # space to hold the result

        for i in range(total_segments):  # for each segment
            current_hop = hop_size * i  # figure out the current segment offset
            segment = proc[current_hop:current_hop + fft_size]  # get the current segment
            windowed = segment * window  # multiply by the half cosine function
            padded = np.append(windowed, inner_pad)  # add 0s to double the length of the data
            spectrum = np.fft.fft(padded) / fft_size  # take the Fourier Transform and scale by the number of samples
            autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
            result[i, :] = autopower[:fft_size]  # append to the results array

        result = 20 * np.log10(result)  # scale to db
        result = np.clip(result, -40, 200)  # clip values
        return result





if __name__ == '__main__':

    filenames = [file for file in glob.glob("/home/spandyie/PycharmProjects/LambwaveComposite/data/*.mat")]

    with open('/home/spandyie/PycharmProjects/LambwaveComposite/data/x_y_data.txt') as xy_label:
        fileN = xy_label.readlines()
        fileN = [f.rstrip().split(" ") for f in fileN]

    y = np.array([int(i[1]) for i in fileN])

    print(y.shape)
    y_one_hot = np.eye(5)[y-1]

    Lamb_wave_data ={}
    f,t, zx ={},{},{}
    for file_n in filenames:
        fileName = file_n.split("/")[-1]
        lamb_wave_object = Ace()
        lamb_wave_object.load(file_n)
        Lamb_wave_data[fileName] = lamb_wave_object
        f[fileName], t[fileName], zx[fileName] = FFT(Lamb_wave_data[fileName]).st_ft


        # the FFT transfrom is saved a 3-D numpy array
    sensor_signal =np.asarray([np.abs(value) for key, value in zx.items()])
    frequency = np.asarray([np.abs(value) for key, value in f.items()])
    time = np.asarray([np.abs(value) for key, value in t.items()])

    #sensor_signal = sensor_signal.reshape((-1,129,30))




    RNN(x=sensor_signal,y=y_one_hot).train()








