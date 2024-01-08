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

path = 'data/SD_data/'

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
    AUC = roc_auc_score(Y_test, x_test_scr)
    return round(AUC, 3)

def mixture_learning(Aggr, minor='SAS', **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')

    train_data, val_data, test_data = Aggr
    X_test, Y_test, R_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=100)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EUR'], x_test_scr[R_test == 'EUR']
    Y_minor, scr_minor = Y_test[R_test == minor], x_test_scr[R_test == minor]
    EA_AUC, AA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_minor, scr_minor)
    A_AUC, EA_AUC, AA_AUC = round(A_AUC, 3),round(EA_AUC, 3), round(AA_AUC, 3)
    return [A_AUC, EA_AUC, AA_AUC]

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
    return round(Naive_AUC, 3)

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
    K.set_value(model.optimizer.lr, tune_lr)
    model.fit(Minor_train_data[0], Minor_train_data[1], validation_data=Minor_val_data, batch_size=batch_size,
              epochs=n_epochs, callbacks=[es], verbose=0)
    prob = np.round(model.predict(Minor_test_X), decimals=3)
    A_AUC = roc_auc_score(Minor_test_Y, prob)
    return round(A_AUC, 3)

def get_data(file, seed=0):
    A = loadmat(file)
    R = A['R']
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
    EAA_X_train, EAA_Y_train = X_train[R_train == 'SAS'], Y_train[R_train == 'SAS']
    EA_X_val, EA_Y_val = X_val[R_val == 'EUR'], Y_val[R_val == 'EUR']
    EAA_X_val, EAA_Y_val = X_val[R_val == 'SAS'], Y_val[R_val == 'SAS']
    EA_X_test, EA_Y_test = X_test[R_test == 'EUR'], Y_test[R_test == 'EUR']
    EAA_X_test, EAA_Y_test = X_test[R_test == 'SAS'], Y_test[R_test == 'SAS']

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

def run_cv(seed):
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
                  'learning_rate': 0.15, 'lr_decay': 0.05, 'dropout': 0.5, 'tune_lr':0.2,
                  'L1_reg': 0.0001, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}

    Aggr, EA, Minor = get_data(path+"EUR_SAS_0.77_h0.5.mat", seed=seed)

    Mix0, Mix1, Mix2 = mixture_learning(Aggr, **para_mix)
    ind_1 = independent_learning(EA, **para_ind_EA)
    ind_2 = independent_learning(Minor, **para_ind_AA)
    naive = naive_transfer(EA, Minor, **para_naive)
    tl2 = super_transfer(EA, Minor, **para_super)
    row = [Mix0, Mix1, Mix2, ind_1, ind_2, naive, tl2]
    print(row, seed)
    return row

def main():
    arguments = sys.argv
    print(arguments)
    res = []
    for seed in range(20):
        row = run_cv(seed)
        res.append(row)
    df_res = pd.DataFrame(res, columns=['Mix0', 'Mix1', 'Mix2', 'Ind1', 'Ind2', 'naive', 'TL2'])
    print(df_res)

if __name__ == '__main__':
    main()
