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
import theano.tensor as T
import theano
import timeit


# from tensorflow import set_random_seed
import numpy as np

# from modelK import get_model, train_and_test

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

seed(11111)
# set_random_seed(11111)
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

def train_test(train, test, seed=0, batch_size=32, k=-1,
               learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9,
               L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):

    X_train, Y_train, R_train, Samples_train = train
    X_test, Y_test, R_test, Samples_test = test
    print(X_train.shape, X_test.shape,(X_test.shape[0]/2, X_test.shape[1]))
    select = SelectKBest(k=k)
    select.fit(X_train, Y_train)
    X_train = select.transform(X_train)
    X_test = select.transform(X_test)
    X_train, Y_train, R_train = add_covar(X_train, Y_train, R_train, Samples_train)
    X_test, Y_test, R_test = add_covar(X_test, Y_test, R_test, Samples_test)

    train_data = (X_train, Y_train)
    df = pd.DataFrame(X_test)
    df['Y']= Y_test
    df['R']= R_test
    df['YR']=df['Y'].map(str) + df['R']
    print(df['R'].value_counts())
    val_samples, test_samples = train_test_split(df, test_size=0.5, random_state=seed, stratify=df['YR'])

    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    X_val = val_samples.values
    val_data = (X_val, Y_val)

    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_test = test_samples.values
    test_data = (X_test, Y_test)

    print('start training')
    model = build_model(n_in=X_train.shape[1],
             learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
             lr_decay=lr_decay, momentum=momentum,
             L2_reg=L2_reg, L1_reg=L1_reg,
             activation="relu",
             dropout=dropout,
             input_dropout=0.25)
    # model = train_and_test(model, train_data, val_data=val_data, n_epochs=n_epochs, batch_size=batch_size)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs, callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test=='EA'], x_test_scr[R_test=='EA']
    Y_AA, scr_AA = Y_test[R_test == 'AA'], x_test_scr[R_test == 'AA']
    EA_AUC, EAA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_AA, scr_AA)
    print(A_AUC, EA_AUC, EAA_AUC)
    return


def transfer_learning(train, test, seed=0, batch_size=32, k=-1,
               learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9,
               L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    Aggr, EA, EAA = prepare_data(train, test, k, seed)
    train_data, val_data, test_data = Aggr
    EA_train_data, EA_val_data, EA_test_data = EA
    EAA_train_data, EAA_val_data, EAA_test_data = EAA

    Map = {}
    print('start training')
    X_train, Y_train = train_data
    X_test, Y_test, R_test = test_data
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout, input_dropout=None)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs, callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EA'], x_test_scr[R_test == 'EA']
    Y_AA, scr_AA = Y_test[R_test == 'AA'], x_test_scr[R_test == 'AA']
    EA_AUC, AA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_AA, scr_AA)
    Map["Aggr"], Map["EA_AUC"], Map["AA_AUC"] = A_AUC, EA_AUC, AA_AUC
    print(Map)
    df_scr = pd.DataFrame()
    df_scr['Aggr_Y']= Y_test
    df_scr['Aggr_scr'] = x_test_scr
    df_scr['Aggr_R'] = R_test

    EA_X_test, EA_Y_test = EA_test_data
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout, input_dropout=None)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs, callbacks=[es], verbose=0)
    EA_test_scr = np.round(model.predict(EA_X_test), decimals=3)
    EA_AUC = roc_auc_score(EA_Y_test, EA_test_scr)
    Map['EA_Ind'] = EA_AUC
    print(Map)
    df1 = pd.DataFrame()
    df1['EA_ind_Y'] =EA_Y_test
    df1['EA_ind_scr'] = EA_test_scr
    df_scr = pd.concat([df_scr, df1], ignore_index=True, axis=1)

    EAA_X_test, EAA_Y_test = EAA_test_data
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
                        lr_decay=lr_decay, momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
                        input_dropout=None)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EAA_train_data[0], EAA_train_data[1], validation_data=EAA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    EAA_test_scr = np.round(model.predict(EAA_X_test), decimals=3)
    EAA_AUC = roc_auc_score(EAA_Y_test, EAA_test_scr)
    Map['AA_Ind'] = EAA_AUC
    print(Map)
    df1 = pd.DataFrame()
    df1['AA_ind_Y'] =EAA_Y_test
    df1['AA_ind_scr'] = EAA_test_scr
    df_scr = pd.concat([df_scr, df1], ignore_index=True, axis=1)

    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
                        lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
                        input_dropout=None)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    EAA_test_scr = np.round(model.predict(EAA_X_test), decimals=3)
    Naive_AUC = roc_auc_score(EAA_Y_test, EAA_test_scr)
    Map['Naive_AUC'] = Naive_AUC
    print(Map)
    df1 = pd.DataFrame()
    df1['Naive_Y'] =EAA_Y_test
    df1['Naive_scr'] = EAA_test_scr
    df_scr = pd.concat([df_scr, df1], ignore_index=True, axis=1)

    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
                        lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
                        input_dropout=None)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    K.set_value(model.optimizer.lr, 0.25)
    model.fit(EAA_train_data[0], EAA_train_data[1], validation_data=EAA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    prob = np.round(model.predict(EAA_X_test), decimals=3)
    A_AUC = roc_auc_score(EAA_Y_test, prob)

    df1 = pd.DataFrame()
    df1['Naive_Y'] = EAA_Y_test
    df1['TL_scr'] = prob
    df_scr = pd.concat([df_scr, df1], ignore_index=True, axis=1)
    df_scr.to_csv('Prostate_PRS.csv')
    Map['TL_AUC'] = A_AUC
    print(Map)
    # df = pd.DataFrame(Map, index=[seed])
    # df.to_csv(str(seed) + '-Prostate.csv')
    return df_scr


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
    return AUC

def mixture_learning(Aggr, minor='Hispanic', **kwds):
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
    Y_EA, scr_EA = Y_test[R_test == 'EA'], x_test_scr[R_test == 'EA']
    Y_minor, scr_minor = Y_test[R_test == minor], x_test_scr[R_test == minor]
    EA_AUC, AA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_minor, scr_minor)
    Map = {}
    Map["Mix0"], Map["Mix1"], Map["Mix2"] = A_AUC, EA_AUC, AA_AUC
    return Map

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
    tune_lr = kwds.pop('tune_lr')
    tune_batch = kwds.pop('tune_batch')
    tune_epochs = kwds.pop('tune_epochs')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test

    model = build_model(n_in=EA_train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=100)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    K.set_value(model.optimizer.lr, tune_lr)
    model.fit(Minor_train_data[0], Minor_train_data[1], validation_data=Minor_val_data, batch_size=tune_batch,
              epochs=tune_epochs, callbacks=[es], verbose=0)
    prob = np.round(model.predict(Minor_test_X), decimals=3)
    A_AUC = roc_auc_score(Minor_test_Y, prob)
    return A_AUC


def get_data(file, seed=0):
    df = pd.read_csv(file, sep=',', index_col=0)
    # df['YR'] = df['Y'].map(str) + df['R']
    df = df[df['R'].isin(['EA', 'Hispanic'])]
    df = df[df['Y'].isin([0,1])]

    train, test = train_test_split(df, test_size=0.8, random_state=0, shuffle=True, stratify=df['YR'])
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

    EA_X_train, EA_Y_train = X_train[R_train == 'EA'], Y_train[R_train == 'EA']
    EAA_X_train, EAA_Y_train = X_train[R_train == 'Hispanic'], Y_train[R_train == 'Hispanic']
    EA_X_val, EA_Y_val = X_val[R_val == 'EA'], Y_val[R_val == 'EA']
    EAA_X_val, EAA_Y_val = X_val[R_val == 'Hispanic'], Y_val[R_val == 'Hispanic']
    EA_X_test, EA_Y_test = X_test[R_test == 'EA'], Y_test[R_test == 'EA']
    EAA_X_test, EAA_Y_test = X_test[R_test == 'Hispanic'], Y_test[R_test == 'Hispanic']

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


def run_cv(seed=0):

    epoch = 100
    para_mix = {'batch_size': 50, 'momentum': 0.9, 'n_epochs': 100,
                'learning_rate': 0.15, 'lr_decay': 0.003, 'dropout': 0.5,
                'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}  # 68,68,57
    para_ind_EA = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.25, 'lr_decay': 0.003, 'dropout': 0.5,
                   'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}
    para_ind_AA = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': epoch,
                   'learning_rate': 0.15, 'lr_decay': 0.003, 'dropout': 0.5,
                   'L1_reg': 0.000, 'L2_reg': 0.0001, 'hidden_layers_sizes': [200]}
    para_naive = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': epoch,
                  'learning_rate': 0.25, 'lr_decay': 0.003, 'dropout': 0.5,
                  'L1_reg': 0.0, 'L2_reg': 0.0001, 'hidden_layers_sizes': [100]}
    para_super = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': epoch, 'tune_batch': 32,
                  'learning_rate': 0.01, 'lr_decay': 0.0003, 'dropout': 0.5, 'tune_lr': 0.0015, 'tune_epochs': 50,
                  'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [200]}

    Aggr, EA, Minor = get_data("EA_Hispanic.csv", seed=seed)
    Map = mixture_learning(Aggr, **para_mix)
    Map['ind_1'] = independent_learning(EA, **para_ind_EA)
    Map['ind_2'] = independent_learning(Minor, **para_ind_AA)
    Map['naive'] = naive_transfer(EA, Minor, **para_naive)
    Map['tl2'] = super_transfer(EA, Minor, **para_super)
    df = pd.DataFrame(Map, index=[seed])
    return df

def main():
    arguments = sys.argv
    print(arguments)
    df_res = pd.DataFrame()
    for seed in range(20):
        df1 = run_cv(seed)
        df_res = df_res.append(df1)
    print(df_res)

if __name__ == '__main__':
    main()
