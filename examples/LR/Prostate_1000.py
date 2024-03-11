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

from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
import theano.tensor as T
import theano
import timeit
from datetime import datetime
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

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

def get_covar():
    df = pd.DataFrame()
    path = 'data/Prostate/Covar/Subject_Phenotypes/'
    for filename in os.listdir(path):
        df1 = pd.read_csv(path + filename, sep='\t', skiprows=10, index_col='SUBJECT_ID',
                          usecols=['SUBJECT_ID', 'AGE', 'FH_PROS'])
        df1 = df1.dropna()
        df1 = df1[df1['AGE'] > 0]
        df1 = df1[df1['FH_PROS'].isin([0, 1])]
        df = df.append(df1)

    path1 = 'data/Prostate/Covar/'
    df_map = pd.read_csv(path1 + 'Prostate_Cancer_Sample.MULTI.txt', sep='\t', skiprows=10,
                         index_col='SUBJECT_ID', usecols=['SUBJECT_ID', 'SAMPLE_ID'])
    df = df.join(df_map, how='left')
    df = df.reset_index().set_index('SAMPLE_ID')
    df = df.drop(columns=['SUBJECT_ID'])

    A = df['AGE'].values
    df['AGE'] = (A - A.mean())/A.std()
    return df

def add_covar(X, Y, R, samples):
    df = get_covar()
    df_data = pd.DataFrame(X, index=samples)
    df_data['Y'] = Y
    df_data['R'] = R
    df_data['YR'] = df_data['Y'].map(str) + df_data['R']
    df_data = df_data.join(df, how='inner')
    Y, R = df_data['Y'].values, df_data['R'].values
    df_data = df_data.drop(columns=['Y', 'R', 'YR'])
    X = df_data.values
    return X, Y, R

def prepare_data(train, test, k, seed):
    X_train, Y_train, R_train, Samples_train = train
    X_test, Y_test, R_test, Samples_test = test
    print(X_train.shape, X_test.shape, (X_test.shape[0] / 2, X_test.shape[1]))
    select = SelectKBest(k=k)
    select.fit(X_train, Y_train)
    X_train = select.transform(X_train)
    X_test = select.transform(X_test)

    X_train, Y_train, R_train = add_covar(X_train, Y_train, R_train, Samples_train)
    X_test, Y_test, R_test = add_covar(X_test, Y_test, R_test, Samples_test)

    df = pd.DataFrame(X_test)
    df['Y'] = Y_test
    df['R'] = R_test
    df['YR'] = df['Y'].map(str) + df['R']

    val_samples, test_samples = train_test_split(df, test_size=0.5, random_state=seed, shuffle=True, stratify=df['YR'])
    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_val, X_test = val_samples.values, test_samples.values

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test, R_test)

    EA_X_train, EA_Y_train = X_train[R_train == 'EA'], Y_train[R_train == 'EA']
    EAA_X_train, EAA_Y_train = X_train[R_train == 'AA'], Y_train[R_train == 'AA']
    EA_X_val, EA_Y_val = X_val[R_val == 'EA'], Y_val[R_val == 'EA']
    EAA_X_val, EAA_Y_val = X_val[R_val == 'AA'], Y_val[R_val == 'AA']
    EA_X_test, EA_Y_test = X_test[R_test == 'EA'], Y_test[R_test == 'EA']
    EAA_X_test, EAA_Y_test = X_test[R_test == 'AA'], Y_test[R_test == 'AA']

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

def mixture_learning(Aggr, minor='AA', **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')

    train_data, val_data, test_data = Aggr
    X_test, Y_test, R_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EA'], x_test_scr[R_test == 'EA']
    Y_minor, scr_minor = Y_test[R_test == minor], x_test_scr[R_test == minor]
    EA_AUC, AA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_minor, scr_minor)
    Map = {}
    Map["Mix0"], Map["Mix1"], Map["Mix2"] = A_AUC, EA_AUC, AA_AUC
    return Map

def independent_learning(data, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    AUC = roc_auc_score(Y_test, x_test_scr)
    return AUC

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
    Naive_AUC = roc_auc_score(Minor_test_Y, Minor_test_scr)
    return Naive_AUC

def super_transfer(EA, Minor, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test

    model = build_model(n_in=EA_train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    K.set_value(model.optimizer.lr, 0.25)
    model.fit(Minor_train_data[0], Minor_train_data[1], validation_data=Minor_val_data, batch_size=batch_size,
              epochs=n_epochs, callbacks=[es], verbose=0)
    prob = np.round(model.predict(Minor_test_X), decimals=3)
    A_AUC = roc_auc_score(Minor_test_Y, prob)
    return A_AUC

def get_data(file):
    A = loadmat(file)
    R = A['R'][0]
    Y = A['Y'][0].astype('int32')
    X = A['X'].astype('float32')
    Samples = A['Samples']
    R = [str(row[0]) for row in R]
    Samples = [s.strip() for s in Samples]
    data = (X, Y, R, Samples)
    return data

def run_cv(seed):

    train = get_data('data/Prostate/0_train_X4.mat')
    test = get_data('data/Prostate/0_test_X4.mat')

    n_epoch=200
    k=1000
    para_mix = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoch,
                     'learning_rate': 0.25, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EA = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoch,
                     'learning_rate': 0.25, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_AA = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoch,
            'learning_rate': 0.25, 'lr_decay': 0.005, 'dropout': 0.5,
            'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_naive = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoch,
                'learning_rate': 0.25, 'lr_decay': 0.005, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_super = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.005, 'dropout': 0.5,
                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}

    res = pd.DataFrame()
    for seed in range(20):
        Aggr, EA, Minor = prepare_data(train, test, k, seed)
        Map = mixture_learning(Aggr, **para_mix)
        Map['ind_1'] = independent_learning(EA, **para_ind_EA)
        Map['ind_2'] = independent_learning(Minor, **para_ind_AA)
        Map['naive'] = naive_transfer(EA, Minor, **para_naive)
        _, EA_s, Minor_t = prepare_data(train, test, k, seed)
        Map['tl2'] = super_transfer(EA_s, Minor_t, **para_super)
        df = pd.DataFrame(Map, index=[seed])
        res = res.append(df)
    return res

def main():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    arguments = sys.argv
    print(arguments)
    seed = int(arguments[1])
    df = run_cv(seed)
    print(df)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

if __name__ == '__main__':
    main()
