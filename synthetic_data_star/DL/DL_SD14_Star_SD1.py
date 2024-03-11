import keras
from keras import backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l1_l2
from numpy.random import seed
import pandas as pd
import random as rn
import os
import sys
import collections
from sklearn.preprocessing import StandardScaler

from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np

from get_performance import get_performance

path = '../../data/SD_data_Star/'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)

def build_model(n_in,
             learning_rate, hidden_layers_sizes=None,
             lr_decay=0.0, momentum=0.9,
             L2_reg=0.0, L1_reg=0.0,
             activation="relu",
             dropout=None,
             input_dropout=None):

    model_input = Input(shape=(n_in,), dtype='float32')

    if hidden_layers_sizes == None:
        z = Dense(1, name='OutputLayer')(model_input)
    else:
        for idx in range(len(hidden_layers_sizes)):
            layer_size = hidden_layers_sizes[idx]
            if idx == 0:
                if input_dropout:
                    input = Dropout(input_dropout)(model_input)
                else:
                    input = model_input
                z = Dense(hidden_layers_sizes[idx], activation=activation, name='HiddenLayer',
                          kernel_initializer=keras.initializers.glorot_uniform(seed=11111),
                          kernel_regularizer=l1_l2(l1=L1_reg, l2=L2_reg), bias_regularizer=l1_l2(l1=L1_reg, l2=L2_reg))(
                    input)
            else:
                z = Dense(layer_size, activation=activation, name='HiddenLayer' + str(idx + 1),
                      kernel_initializer=keras.initializers.glorot_uniform(seed=11111),
                      kernel_regularizer=l1_l2(l1=L1_reg, l2=L2_reg), bias_regularizer=l1_l2(l1=L1_reg, l2=L2_reg))(z)
            if dropout:
                z = Dropout(rate=dropout)(z)
        z = Dense(1, name='OutputLayer')(z)

    model_output = Activation('sigmoid')(z)
    model = Model(model_input, model_output)

    if momentum:
        sgd = SGD(lr=learning_rate, decay=lr_decay, momentum=momentum, nesterov=True)
    else:
        sgd = SGD(lr=learning_rate, decay=lr_decay)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model

def make_df(Y_test, R, x_test_scr):
    df = pd.DataFrame()
    df['Y'] = Y_test
    df['R'] = R
    df['Scr'] = x_test_scr
    return df

def independent_learning(data, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=100)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)

    x_test_scr = np.round(model.predict(X_test), decimals=3)
    df = make_df(Y_test, "", x_test_scr)
    return df

def mixture_learning(Aggr, minor='AMR', **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')

    train_data, val_data, test_data = Aggr
    X_test, Y_test, R_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=100)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    df = make_df(Y_test, R_test, x_test_scr)
    return df

def naive_transfer(EA, Minor, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test

    model = build_model(n_in=EA_train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    Minor_test_scr = np.round(model.predict(Minor_test_X), decimals=3)
    df = make_df(Minor_test_Y, "", Minor_test_scr)
    return df

def super_transfer(EA, Minor, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    tune_lr = kwds.pop('tune_lr')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test

    model = build_model(n_in=EA_train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=100)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    K.set_value(model.optimizer.lr, tune_lr) #0.2
    model.fit(Minor_train_data[0], Minor_train_data[1], validation_data=Minor_val_data, batch_size=batch_size,
              epochs=n_epochs, callbacks=[es], verbose=0)
    prob = np.round(model.predict(Minor_test_X), decimals=3)
    df = make_df(Minor_test_Y, "", prob)
    return df

def get_data(file, seed=0, minor="AFR"):
    A = loadmat(file)
    R = A['R'][0]
    R = [row[0] for row in R]
    Y = A['Y'][0].astype('int32')
    X = A['X'].astype('float32')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame(X)
    df['R'], df['Y'] = R, Y
    df['YR'] = df['Y'].map(str) + df['R']

    train, test = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True, stratify=df['YR'])
    Y_train, R_train = train['Y'].values, train['R'].values
    train = train.drop(columns=['Y', 'R', 'YR'])
    X_train = train.values

    val_samples, test_samples = train_test_split(test, test_size=0.5, random_state=seed, shuffle=True, stratify=test['YR'])
    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_val, X_test = val_samples.values, test_samples.values

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test, R_test)

    EA_X_train, EA_Y_train = X_train[R_train == 'EUR'], Y_train[R_train == 'EUR']
    EAA_X_train, EAA_Y_train = X_train[R_train == minor], Y_train[R_train == minor]
    EA_X_val, EA_Y_val = X_val[R_val == 'EUR'], Y_val[R_val == 'EUR']
    EAA_X_val, EAA_Y_val = X_val[R_val == minor], Y_val[R_val == minor]
    EA_X_test, EA_Y_test = X_test[R_test == 'EUR'], Y_test[R_test == 'EUR']
    EAA_X_test, EAA_Y_test = X_test[R_test == minor], Y_test[R_test == minor]

    EA_train_data = (EA_X_train, EA_Y_train)
    EA_val_data = (EA_X_val, EA_Y_val)
    EA_test_data = (EA_X_test, EA_Y_test)

    EAA_train_data = (EAA_X_train, EAA_Y_train)
    EAA_val_data = (EAA_X_val, EAA_Y_val)
    EAA_test_data = (EAA_X_test, EAA_Y_test)

    Aggr = [train_data, val_data, test_data]
    EA = [EA_train_data, EA_val_data, EA_test_data]
    EAA = [EAA_train_data, EAA_val_data, EAA_test_data]

    return [Aggr, EA, EAA]

Map_SD = { 1: "EUR_AMR_0.8_h0.5.mat", 2: "EUR_SAS_0.77_h0.5.mat", 3:"EUR_EAS_0.58_h0.5.mat", 4:"EUR_AFR_0.54_h0.5.mat",
        5:"EUR_AMR_0.8_h0.25.mat", 6:"EUR_SAS_0.77_h0.25.mat", 7:"EUR_EAS_0.58_h0.25.mat", 8:"EUR_AFR_0.54_h0.25.mat",
        9:"EUR_AMR_0.8_h0.5.mat", 10:"EUR_SAS_0.74_h0.5.mat", 11:"EUR_EAS_0.42_h0.5.mat", 12:"EUR_AFR_0.36_h0.5.mat",
        13:"EUR_AMR_0.8_h0.25.mat", 14:"EUR_SAS_0.74_h0.25.mat", 15:"EUR_EAS_0.42_h0.25.mat", 16:"EUR_AFR_0.36_h0.25.mat"
}

Map_Minor = ['AMR', 'SAS', 'EAS', 'AFR']*4

def init_paras():
    ################# Paras SD1 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}

    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.2,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    Para_SD1 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD2 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.2,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    Para_SD2 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD3 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.2,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    Para_SD3 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD4 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch, #'tune_batch': 100, 'tune_epochs': epoch,
                  'learning_rate': 0.15, 'lr_decay': 0.03, 'dropout': 0.5, 'tune_lr': 0.15,
                  'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD4 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD5 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.02,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    Para_SD5 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD6 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.02,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    Para_SD6 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD7 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.2, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.04, 'dropout': 0.5, 'tune_lr': 0.01,
                  'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD7 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD8 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.1, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.2, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': 50, #'tune_batch': 100,
                  'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.015, #'tune_epochs': 100,
                  'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD8 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD10 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.02, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.01, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.02,
                  'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD10 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD11 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.02, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.01, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.15,
                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD11 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD12 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.02, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.01, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.7, 'n_epochs': 100, #'tune_batch': 50,
                  'learning_rate': 0.015, 'lr_decay': 0.00, 'dropout': 0.6, 'tune_lr': 0.01, #'tune_epochs': 200,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    Para_SD12 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD14 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.02, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.01, 'lr_decay': 0.03, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.15,
                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD14 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD15 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.6, 'n_epochs': epoch,
                   'learning_rate': 0.02, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.5, 'n_epochs': epoch,
                  'learning_rate': 0.02, 'lr_decay': 0.05, 'dropout': 0.5,
                  'L1_reg': 0.0001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.15,
                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD15 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    ################# Paras SD16 ###########################
    epoch = 100
    para_mix = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 200, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.003, 'dropout': 0.5,
                   'L1_reg': 0.000, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.003, 'dropout': 0.5,
                  'L1_reg': 0.0, 'L2_reg': 0.0001, 'hidden_layers_sizes': [100]}
    para_super = {'batch_size': 100, 'momentum': 0.9, 'n_epochs': epoch, #'tune_batch': 100,
                  'learning_rate': 0.25, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr': 0.15, #'tune_epochs': epoch,
                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    Para_SD16 = [para_mix, para_ind_EA, para_ind_AA, para_naive, para_super]

    Para_SD = [Para_SD1, Para_SD2, Para_SD3, Para_SD4, Para_SD5, Para_SD6, Para_SD7, Para_SD8,
               Para_SD1, Para_SD10, Para_SD11, Para_SD12, Para_SD5, Para_SD14, Para_SD15, Para_SD16]
    return Para_SD

Para_SD = init_paras()

def run_sd(SD, metric='auc'):

    para_mix, para_ind_EA, para_ind_AA, para_naive, para_super = Para_SD[SD-1]
    res = []
    for seed in range(20):

        Aggr, EA, Minor = get_data(path + Map_SD[SD], seed=seed, minor=Map_Minor[SD-1])
        df_mix0 = mixture_learning(Aggr, **para_mix)
        df_mix1 = df_mix0[df_mix0['R'] == 'EUR']
        df_mix2 = df_mix0[df_mix0['R'] == Map_Minor[SD-1]]
        df_ind_1 = independent_learning(EA, **para_ind_EA)
        df_ind_2 = independent_learning(Minor, **para_ind_AA)
        df_ind_2['R'] = Map_Minor[SD-1]
        df_naive = naive_transfer(EA, Minor, **para_naive)
        df_tl_2 = super_transfer(EA, Minor, **para_super)

        mix0 = get_performance(df_mix0['Y'].values, df_mix0['Scr'].values, metric=metric)
        mix1 = get_performance(df_mix1['Y'].values, df_mix1['Scr'].values, metric=metric)
        mix2 = get_performance(df_mix2['Y'].values, df_mix2['Scr'].values, metric=metric)
        ind1 = get_performance(df_ind_1['Y'].values, df_ind_1['Scr'].values, metric=metric)
        ind2 = get_performance(df_ind_2['Y'].values, df_ind_2['Scr'].values, metric=metric)
        naiv = get_performance(df_naive['Y'].values, df_naive['Scr'].values, metric=metric)
        tl_2 = get_performance(df_tl_2['Y'].values, df_tl_2['Scr'].values, metric=metric)
        row = [mix0, mix1, mix2, ind1, ind2, naiv, tl_2]

        print (seed, row)
        res.append(row)
        print("saving iteration ", seed)

    df_res = pd.DataFrame(res, columns=['Mix0', 'Mix1', 'Mix2', 'Ind1', 'Ind2', 'naive', 'TL2'])
    df_res['SD'] = SD
    print(df_res)

SData = [1,2,3,4,5,6,7,8,10,11,12,14,15,16]

def main():
    arguments = sys.argv
    print(arguments)
    for SD in SData:
        print("Working on SD ", SD)
        df = run_sd(SD)

if __name__ == '__main__':
    main()
