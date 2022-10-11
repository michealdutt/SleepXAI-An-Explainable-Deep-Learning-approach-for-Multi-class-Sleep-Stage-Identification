from keras import optimizers, losses, activations, models, backend
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM, Reshape, Flatten
import numpy as np
from keras_contrib.layers import CRF
from utils import WINDOW_SIZE

def encode_model():    
    inp = Input(shape=(3000,1))
    conv = Convolution1D(128, kernel_size=10, activation=activations.selu, padding="valid")(inp) 
    conv = Convolution1D(128, kernel_size=10, activation=activations.selu, padding="valid")(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = SpatialDropout1D(rate=0.01)(conv)
    conv = Convolution1D(64, kernel_size=10, activation=activations.selu, padding="valid")(conv) 
    conv = Convolution1D(64, kernel_size=10, activation=activations.selu, padding="valid")(conv)
    conv = MaxPool1D(pool_size=2)(conv)
    conv = SpatialDropout1D(rate=0.01)(conv)
    conv = Convolution1D(32, kernel_size=10, activation=activations.selu, padding="valid")(conv) 
    conv = Convolution1D(32, kernel_size=10, activation=activations.selu, padding="valid")(conv)
    conv = GlobalMaxPool1D()(conv)
    conv = Dropout(rate=0.01)(conv)
    dense_1 = Dropout(0.01)(Dense(64, activation=activations.selu, name="dense_1")(conv))

    epoch_encoder = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adamax(0.001)

    epoch_encoder.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    epoch_encoder.summary()
    return epoch_encoder


#CNN model
def get_model_cnn():
    seq_input = Input(shape=(None, 3000, 1))
    epoch_encoder = encode_model()
    encoded_classifier = TimeDistributed(epoch_encoder)(seq_input)
    encoded_classifier = Convolution1D(128, kernel_size=3, activation="selu", padding="same")(encoded_classifier)
    encoded_classifier = SpatialDropout1D(rate=0.01)(encoded_classifier)
    encoded_classifier = Convolution1D(128, kernel_size=3, activation="selu", padding="same")(encoded_classifier)
    encoded_classifier = Dropout(rate=0.05)(encoded_classifier)
    out = Convolution1D(5, kernel_size=4, activation="softmax", padding="same")(encoded_classifier)
    model = models.Model(seq_input, out)
    model.compile(optimizers.Adamax(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()

    return model



#CRF model
def get_model_cnn_crf(lr=0.001):
    nclass = 5
    seq_input = Input(shape=(None, 3000, 1))
    epoch_encoder = encode_model()
    encoded_classifier = TimeDistributed(epoch_encoder)(seq_input)
    encoded_classifier = Convolution1D(128, kernel_size=3, activation="selu", padding="same")(encoded_classifier)
    encoded_classifier = SpatialDropout1D(rate=0.01)(encoded_classifier)
    encoded_classifier = Convolution1D(128, kernel_size=3, activation="linear", padding="same")(encoded_classifier)
    encoded_classifier = Dropout(rate=0.05)(encoded_classifier)
    # encoded_sequence = Dropout(rate=0.05)(Convolution1D(128, kernel_size=3, activation="tanh", padding="same")(encoded_sequence))
    crf = CRF(nclass, sparse_target=True)
    out = crf(encoded_classifier)
    model = models.Model(seq_input, out)
    model.compile(optimizers.Adam(lr), crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    return model


