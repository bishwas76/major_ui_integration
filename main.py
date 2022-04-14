from fastapi import FastAPI, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import soundfile as sf
import librosa

import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from python_speech_features import mfcc
import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Lambda)
import scipy.io.wavfile as wav

def int_sequence_to_text(int_sequence):
    """ Convert an integer sequence to text """
    """
    Defines two dictionaries for converting 
    between text and integer sequences.
    """
    char_map_str = """
    <SPACE> 0
    ऀ 1
    ँ 2
    ं 3
    ः 4
    अ 5
    आ 6
    इ 7
    ई 8
    उ 9
    ऊ 10
    ऋ 11
    ए 12
    ऐ 13
    ओ 14
    औ 15
    क 16
    ख 17
    ग 18
    घ 19
    ङ 20
    च 21
    छ 22
    ज 23
    झ 24
    ञ 25
    ट 26
    ठ 27
    ड 28
    ढ 29
    ण 30
    त 31
    थ 32
    द 33
    ध 34
    न 35
    प 36
    फ 37
    ब 38
    भ 39
    म 40
    य 41
    र 42
    ल 43
    व 44
    श 45
    ष 46
    स 47
    ह 48
    ऺ 49
    ़ 50
    ा 51
    ि 52
    ी 53
    ु 54
    ू 55
    ृ 56
    ॄ 57
    े 58
    ै 59
    ॉ 60
    ॊ 61
    ो 62
    ौ 63
    ् 64
    ॐ 65
    ॑ 66
    ॒ 67
    ॠ 68
    : 69
    ० 70
    १ 71
    २ 72
    ३ 73
    ४ 74
    ५ 75
    ६ 76
    ७ 77
    ८ 78
    ९ 79
    ॅ 80
    \u200d 81
    \u200c 82
    ! 83
    ? 84
    % 85
    \u200e 86
    ऱ 87
    . 88
    \u200f 89
    \ufeff 90
    फ़ 91
    """
    char_map = {}
    index_map = {}
    for line in char_map_str.strip().split('\n'):
        ch, index = line.split()
        char_map[ch] = int(index)
        index_map[int(index)+1] = ch
    index_map[1] = ' '

    text = []
    for c in int_sequence:        
        if c==0:
            ch=' '
        else:
            ch = index_map[c]
        text.append(ch)
    return text
    

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def final_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=93, dropout_rate=0.5, number_of_layers=2, 
    cell=GRU, activation='tanh'):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='layer_1_conv',
                     dilation_rate=1)(input_data)
    #maxpool = MaxPooling1D(pool_size=2, strides=1, padding='same')(conv_1d)
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate, reset_after=False)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate, reset_after=False)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate, reset_after=False)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn', reset_after=False)(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    return model
    
def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1

def prediction_func(audio_loc, model, model_path, window=20, max_freq=8000):
    
    (rate,sig) = wav.read(audio_loc)
    print(audio_loc)
    mfcc_feat = mfcc(sig,rate, nfft=1024)
    feat_dim = calc_feat_dim(window, max_freq)
    feats_mean = np.zeros((feat_dim,))
    feats_std = np.ones((feat_dim,))
    eps=1e-14

    feats = np.vstack(mfcc_feat)
    feats_mean = np.mean(feats, axis=0)
    feats_std = np.std(feats, axis=0)
    data_point = (feats - feats_mean) / (feats_std + eps)

    model.load_weights(model_path)
    prediction = model.predict(np.expand_dims(data_point, axis=0))
    output_length = [model.output_length(data_point.shape[0])] 
    pred_ints = (K.eval(K.ctc_decode(
                prediction, output_length)[0][0])+1).flatten().tolist()
    return ''.join(int_sequence_to_text(pred_ints))

model_end = final_model(input_dim=13,
                        filters=220,
                        kernel_size=11, 
                        conv_stride=2,
                        conv_border_mode='valid',
                        units=400,
                        activation='relu',
                        cell=GRU,
                        dropout_rate=0.5,
                        number_of_layers=2)

@app.post('/predict')
async def predict(file: UploadFile):
    myFile = file.file
    content = myFile.read()
    async with aiofiles.open(r'.\audio.wav', 'wb') as out_file:
        await out_file.write(content)
    sound, sr = librosa.load(r'.\audio.wav')
    sf.write(r'.\audio.wav', sound, sr)
    predictedText = prediction_func(audio_loc=r'.\audio.wav',model=model_end,model_path=".\CNN_GRU_GRU_normh5.h5")
    responseData = {
        "predictedText": predictedText,
    }
    return json.dumps(responseData, ensure_ascii=False).encode('utf-8')

