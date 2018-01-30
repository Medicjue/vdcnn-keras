# -*- coding: utf-8 -*-
import vdcnn
import pandas as pd
from utils import CharEmbeddedEncoder
import numpy as np
from datetime import datetime as dt

from sklearn.metrics import confusion_matrix

def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y



def main():
    s = dt.now()
    train_data = pd.read_csv('data/ag_news_csv/train.csv', index_col=False, header=None, names=['class', 'content'])
    test_data = pd.read_csv('data/ag_news_csv/test.csv', index_col=False, header=None, names=['class', 'content'])
    f = open('data/ag_news_csv/classes.txt', 'r')
    classes = [i.replace('\n','') for i in list(f)]
    num_classes = len(classes)
    f.close()
    encoder = CharEmbeddedEncoder(n_jobs=4)
    train_X = encoder.transform(train_data['content'].as_matrix())
    train_data['class'] = train_data['class']-1
    train_Y = train_data['class'].as_matrix()
    train_Y = to_categorical(train_Y, nb_classes=num_classes)
    
    test_X = encoder.transform(test_data['content'].as_matrix())
    test_data['class'] = test_data['class']-1
    test_Y = test_data['class'].as_matrix()
    e = dt.now()
    p = e - s
    print('Prepare Data consume:{}'.format(p))
    
    model = vdcnn.create_model(num_classes=num_classes)
    model.summary()
    
    s = dt.now()
    model.fit(train_X, train_Y, epochs=15, batch_size=128, validation_split=0.1, shuffle=True)
    e = dt.now()
    p = e - s
    print('Training Model consume:{}'.format(p))
    
    s = dt.now()
    predict_Y = model.predict_classes(test_X)
    e = dt.now()
    p = e - s
    print('Predict consume:{}'.format(p))
    
    print(confusion_matrix(test_Y, predict_Y))
    
if __name__ == '__main__':
    main()