import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
from keras.layers import Dropout, Dense, TimeDistributed
from keras.models import Sequential
from keras.utils import np_utils
    # keras.utils.np_utils.to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import librosa.display
import pandas as pd
from utils.signal import get_segment, plot_signal, plot_spectra, stft_audio,calc_mfcc, get_segment_stats, envelope, get_total_segment_stats, get_total_audio_stats
from utils.annotation import get_meta_annotation, get_simplified_meta_annotation, get_simplified_annotation
import IPython.display as ipd
from utils.randomforest import run_randomforest
from utils.features import get_boxplot_by_species, get_scree_plot
import seaborn as sns
from hidden_Markov_Model import *
import matplotlib.pyplot as plt

FILE_PATH  = './dryad/annotation_Files/Recording_1/Recording_1_Segment_04.Table.1.selections.txt'
r1_annotation = get_simplified_annotation(FILE_PATH, filter=True, overlap_percent=0.1)
r1, Fs = librosa.load('./dryad/wav_Files/Recording_1/Recording_1_Segment_04.wav')
r1 = get_segment(signal=r1, annotation=r1_annotation,Fs=Fs)
r1_0 = r1[0]
pcp = get_segment_stats(signal=r1_0, Fs=Fs, frame_length=512,
                        hop_length=256,mfcc_order=0,
                        annotation=r1_annotation[0:1],
                        mean=False,filter=True)
pcp = pd.DataFrame(columns=pcp.columns)
AUDIO_PATH = './dryad/wav_Files'
total_pcp = get_total_audio_stats(audio_path='./dryad/wav_Files',
                                  mfcc_order=0, frame_length=512, hop_length=256,
                                  pcp=pcp,
                                  env=False, thres=0.0005,
                                  mean=False,signal_filter=True,
                                  annotation_filter=True, overlap_percent=0.1)
final_species = total_pcp[['species','segment']].groupby('species').size().sort_values(ascending=False).head(10).reset_index()['species'].values
total_pcp = total_pcp[total_pcp['species'].isin(final_species)]

def build_rand_feat(pcp, species_to_int_dict):
    X = []
    y = []
    segment_list = pcp['segment'].drop_duplicates().to_list()
    n_seg = len(segment_list)
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_seg)):
        selected_segment = np.random.choice(segment_list) # 抽一个乐器
        selected_pcp = pcp[pcp['segment']==selected_segment]
        X_sample = selected_pcp.drop(columns=['segment','species'])
        _min = min(np.amin(X_sample),_min)
        _max = max(np.amax(X_sample),_max)
        X.append(X_sample if config.mode=='conv' else X_sample.T)
        y.append(species_to_int_dict[str(selected_pcp['species'].drop_duplicates().values)]) # classes 是一个map classes to int的list,比如EATO的index=1，label=选取文件的classes
    X, y = np.array(X), np.array(y)
    X = (X - _min)/(_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y =  np_utils.to_categorical(y, num_classes=10) # num_classes：乐器数
    return X,y

def get_conv_model():
    model=Sequential()
    model.add(Conv2D(16, (5,5), activation='relu',strides=(2,2),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(32, (5,5), activation='relu',strides=(2,2),
                     padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', strides=(2, 2),
                     padding='same', input_shape=input_shape))
    model.add(Conv2D(128, (5, 5), activation='relu', strides=(2, 2),
                     padding='same'))
    model.add(MaxPool2D(2,2))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    return model

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, rate=32000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.rate = rate
        self.nfft = nfft
        self.step= int(rate/10) # frame_length?

config = Config(mode='conv')
species_list = total_pcp['species'].drop_duplicates().to_list()
species_to_int_dict = {y:x for y,x in zip(species_list,range(len(species_list)))}

if config.mode == 'conv':
    X, y = build_rand_feat(pcp=total_pcp,species_to_int_dict=species_to_int_dict)
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2], 1)
    model = get_conv_model()
"""
elif config.mode == 'time':
    X, y = build_rand_feat()
    y_flat = np.argmax(y, axis=1)
    input_shape = (X.shape[1], X.shape[2])
    model = get_recurrent_model()
"""
class_weight=compute_class_weight('balanced',
                                  np.unique(y_flat),
                                  y_flat)

model.fit(X,y, epochs=10, batch=32, shuffle=True, class_weight=class_weight)
