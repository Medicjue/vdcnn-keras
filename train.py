# -*- coding: utf-8 -*-
import vdcnn
import pandas as pd
from utils import CharEmbeddedEncoder
import numpy as np

def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y



def main():
    train_data = pd.read_csv('data/ag_news_csv/train.csv', index_col=False, header=None, names=['class', 'content'])
    f = open('data/ag_news_csv/classes.txt', 'r')
    classes = [i.replace('\n','') for i in list(f)]
    num_classes = len(classes)
    f.close()
    encoder = CharEmbeddedEncoder(n_jobs=4)
    X = encoder.transform(train_data['content'].as_matrix())
    train_data['class'] = train_data['class']-1
    Y = train_data['class'].as_matrix()
    Y = to_categorical(Y, nb_classes=num_classes)
    model = vdcnn.create_model(num_classes=num_classes)
    
    model.summary()
    model.fit(X, Y, epochs=1, batch_size=16, validation_split=0.1, shuffle=True)
    
if __name__ == '__main__':
    main()