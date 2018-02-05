# -*- coding: utf-8 -*-
import vdcnn
import pandas as pd
from utils import CharEmbeddedEncoder
import numpy as np
from datetime import datetime as dt
import random
from sklearn.metrics import confusion_matrix
import gc

import matplotlib.pyplot as plt


def to_categorical(y, nb_classes=None):
    y = np.asarray(y, dtype='int32')

    if not nb_classes:
        nb_classes = np.max(y) + 1

    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.

    return Y



def main():
    random_seed = 23
    epochs = 20
    batch_size = 128
    s = dt.now()
    train_data = pd.read_csv('data/ag_news_csv/train.csv', index_col=False, header=None, names=['class', 'title', 'content'])
    test_data = pd.read_csv('data/ag_news_csv/test.csv', index_col=False, header=None, names=['class', 'title', 'content'])
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
    
    model = vdcnn.create_model(num_classes=num_classes, input_dim=encoder.char_dict_len)
    model.summary()
    
    
    losses = []
    accuracies = []
    for epoch in range(epochs):
        
        shuffle_index = [i for i in range(len(train_X))]
        random.seed = random_seed + epoch
        random.shuffle(shuffle_index)
        
        tmp = []
        for i in range(len(train_X)):
            tmp.append(train_X[shuffle_index[i]])
        train_X = np.asarray(tmp)
        
        tmp = []
        for i in range(len(train_Y)):
            tmp.append(train_Y[shuffle_index[i]])
        train_Y = np.asarray(tmp)
        del(tmp)
        
        loss_of_epoch = []
        acc_of_epoch = []
        for start_index in range(0, len(train_X), batch_size):
            end_index = start_index + batch_size
            if end_index > len(train_X):
                end_index = len(train_X)
            batch_train_X = train_X[start_index: end_index]
            batch_train_Y = train_Y[start_index: end_index]
            rtn = model.train_on_batch(x=batch_train_X, y=batch_train_Y)
            loss = rtn[0]
            acc = rtn[1]
            loss_of_epoch.append(loss)
            acc_of_epoch.append(acc)
            del(batch_train_X)
            del(batch_train_Y)
            gc.collect()
        loss_of_epoch = np.mean(loss_of_epoch)
        acc_of_epoch = np.mean(acc_of_epoch)
        losses.append(loss_of_epoch)
        accuracies.append(acc_of_epoch)
        print('Epoch {} completed, loss - {}, acc - {}'.format(epoch+1, loss_of_epoch, acc_of_epoch))
#    s = dt.now()
#    model.fit(train_X, train_Y, epochs=10, batch_size=128, validation_split=0.05, shuffle=True)
    e = dt.now()
    p = e - s
    print('Training Model consume:{}'.format(p))
    
    model.save('model/ag_news_shuffle.mdl')
    
    s = dt.now()
    predict_Y = model.predict_classes(test_X)
    e = dt.now()
    p = e - s
    print('Predict consume:{}'.format(p))
    
    print(confusion_matrix(test_Y, predict_Y))
    pd.DataFrame(confusion_matrix(test_Y, predict_Y)).to_csv('performance/ag_news_shuffle.csv')
    
    plt.plot(losses)
    plt.savefig('figure/loss.png')
    plt.close('all')
    plt.clf()
    
    plt.plot(accuracies)
    plt.savefig('figure/accuracy.png')
    plt.close('all')
    plt.clf()
    
if __name__ == '__main__':
    main()