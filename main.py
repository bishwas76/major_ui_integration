from fastapi import FastAPI, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import aiofiles
import soundfile as sf
import librosa

app = FastAPI()

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
    labels_df = pd.read_csv("index.csv", encoding="utf-8")
    char_map = {}
    index_map = {}
    count = 0
    for i in labels_df["character"]:
        char_map[i] = count
        index_map[count] = i
        count +=1
    index_map[42] = ' '
    index_map
    text = []
    for c in int_sequence:
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
    conv_border_mode, units, output_dim=59, dropout_rate=0.2, number_of_layers=2, 
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
    conv_bn = BatchNormalization(name='conv_batch_norm')(conv_1d)


    if number_of_layers == 1:
        layer = cell(units, activation=activation,
            return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)
    else:
        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='rnn_1', dropout=dropout_rate)(conv_bn)
        layer = BatchNormalization(name='bt_rnn_1')(layer)

        for i in range(number_of_layers - 2):
            layer = cell(units, activation=activation,
                        return_sequences=True, implementation=2, name='rnn_{}'.format(i+2), dropout=dropout_rate)(layer)
            layer = BatchNormalization(name='bt_rnn_{}'.format(i+2))(layer)

        layer = cell(units, activation=activation,
                    return_sequences=True, implementation=2, name='final_layer_of_rnn')(layer)
        layer = BatchNormalization(name='bt_rnn_final')(layer)
    

    time_dense = TimeDistributed(Dense(output_dim))(layer)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
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
                            filters=200,
                            kernel_size=11, 
                            conv_stride=2,
                            conv_border_mode='valid',
                            units=250,
                            activation='tanh',
                            cell=GRU,
                            dropout_rate=0.5,
                            number_of_layers=2)

@app.post('/predict')
async def predict(file: UploadFile):
    myFile = file.file
    content = myFile.read()
    async with aiofiles.open(r'C:\Users\bayer\Desktop\New Folder\audio.wav', 'wb') as out_file:
        await out_file.write(content)
    sound, sr = librosa.load(r'C:\Users\bayer\Desktop\New Folder\audio.wav')
    sf.write(r'C:\Users\bayer\Desktop\New Folder\audio.wav', sound, sr)
    predictedText = prediction_func(audio_loc=r'C:\Users\bayer\Desktop\New Folder\audio.wav',model=model_end,model_path="model_end.h5")
    responceData = {
        'predictedText': predictedText,
    }
    return responceData

