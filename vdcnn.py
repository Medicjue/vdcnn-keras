# -*- coding: utf-8 -*-
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, BatchNormalization, Activation, Dense, Lambda
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD

def create_model(num_classes, num_filters=[64, 128, 256, 512], top_k=3, learning_rate=0.01):
    model = Sequential()
    model.add(Embedding(input_dim=69, output_dim=16, input_length=1014, name='input_embedding'))
    model.add(Conv1D(filters=64, kernel_size=3, strides=2, padding="same"))
    
    for i in range(len(num_filters)):
        conv_filter= num_filters[i]
        
        conv_block = Sequential()
        conv_block.add(Conv1D(filters=conv_filter, 
                              kernel_size=3, 
                              strides=1, 
                              padding='same', 
                              input_shape=list(model.get_output_shape_at(0))[1:]))
        conv_block.add(BatchNormalization())
        conv_block.add(Activation('relu'))
        conv_block.add(Conv1D(filters=conv_filter, 
                              kernel_size=3, 
                              strides=1, 
                              padding='same'))
        conv_block.add(BatchNormalization())
        conv_block.add(Activation('relu'))
        
        model.add(conv_block)
        model.add(MaxPooling1D(pool_size=3, strides=2, padding="same"))
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))
    model.add(Lambda(_top_k, output_shape=(num_filters[-1] * top_k,)))
    model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(2048, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
    model.summary()
    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    return model    
