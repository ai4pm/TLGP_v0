import keras
from keras import backend as K
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l1_l2
from numpy.random import seed
from keras import layers
from keras import regularizers
from datetime import datetime

import pandas as pd
import random as rn
import os
import sys
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from tensorflow import set_random_seed
# import theano.tensor as T
# import theano
import timeit
import numpy as np
import argparse
# sys.path.append('/lustre/haven/user/ygao45/GWAS/')
# from dann_tcga.run import run_dann

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

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

def prepare_data(train, test, seed=0, k=1000, minor='EAA'):
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
    val_samples, test_samples = train_test_split(df, test_size=0.5, random_state=seed, stratify=df['YR'])
    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_val, X_test = val_samples.values, test_samples.values

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test, R_test)
    Agg = [train_data, val_data, test_data]

    EA_X_train, EA_Y_train = X_train[R_train == 'EA'], Y_train[R_train == 'EA']
    EA_X_val, EA_Y_val = X_val[R_val == 'EA'], Y_val[R_val == 'EA']
    EA_X_test, EA_Y_test = X_test[R_test == 'EA'], Y_test[R_test == 'EA']

    EAA_X_train, EAA_Y_train = X_train[R_train == minor], Y_train[R_train == minor]
    EAA_X_val, EAA_Y_val = X_val[R_val == minor], Y_val[R_val == minor]
    EAA_X_test, EAA_Y_test = X_test[R_test == minor], Y_test[R_test == minor]

    EA_train_data = [EA_X_train, EA_Y_train]
    EA_val_data = [EA_X_val, EA_Y_val]
    EA_test_data = [EA_X_test, EA_Y_test]
    EA = [EA_train_data, EA_val_data, EA_test_data]

    EAA_train_data = [EAA_X_train, EAA_Y_train]
    EAA_val_data = [EAA_X_val, EAA_Y_val]
    EAA_test_data = [EAA_X_test, EAA_Y_test]
    EAA = [EAA_train_data, EAA_val_data, EAA_test_data]
    return [Agg, EA, EAA]

def add_covar(X, Y, R, samples):
    path = 'Covar/'
    df1 = pd.read_csv(path + 'c1_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df2 = pd.read_csv(path + 'c3_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                   usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df3 = pd.read_csv(path + 'c6_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df4 = pd.read_csv(path + 'c8_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df5 = pd.read_csv(path + 'c10_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df6 = pd.read_csv(path + 'c14_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])

    df = pd.concat([df1, df2, df3, df4, df5, df6])
    df_map = pd.read_csv(path + 'c1.csv', sep='\t', skiprows=10, index_col='subject_ID', usecols=['subject_ID', 'sample_ID'])
    df = df.join(df_map, how='left')
    df = df.reset_index().set_index('sample_ID')
    df = df.drop(columns=['subject_ID'])

    df_sex_dummy = pd.get_dummies(df['sex'], prefix='sex', drop_first=True)
    df_typesmk_dummy = pd.get_dummies(df['typesmok'], prefix='typesmk', drop_first=True)
    df = df.drop(columns=['sex', 'typesmok'])
    df1 = df.join(df_sex_dummy, how='left')
    df1 = df1.join(df_typesmk_dummy, how='left')


    ## for transfer learning EAA group
    # df1 = df1[df1['packyear_category']>=0]
    # A = df1['age_category'].values
    # df1['age_category'] = (A - A.mean())/A.std()
    # A = df1['packyear_category'].values
    # df1['packyear_category'] = (A - A.mean()) / A.std()
    # print(df1.head())
    # df = df[df['sex']>0]
    # df1 = df1[df1['age_category'] > 7]
    # df1 = df1.drop(columns='age_category')
    # df1['packyear'] = 0
    # df1.loc[df1['packyear']>0, 'packyear'] = 1
    # df1 = df1.drop(columns='packyear_category')
    # df1['typesmk'] = 0
    # df1.loc[df1['typesmok'].isin([2,3]), 'typesmk'] = 1
    # df1 = df1.drop(columns='typesmok')
    # print(df1.head())

    df_data = pd.DataFrame(X, index=samples)
    df_data['Y'] = Y
    df_data['R'] = R
    df_data = df_data.join(df1, how='inner')
    Y, R = df_data['Y'].values, df_data['R'].values
    df_data = df_data.drop(columns=['Y', 'R'])
    X = df_data.values

    # check value counts
    df1 = pd.DataFrame()#[Y,R], columns=['Y', 'R'])
    df1['Y'], df1['R'] = Y,R
    df1['YR'] = df1['Y'].astype(str) + df1['R']
    # print(df1['YR'].value_counts())

    return X, Y, R


def train_test(train, test, batch_size=32, k=-1,
               learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9,
               L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):

    X_train, Y_train, R_train, Samples_train = train
    X_test, Y_test, R_test, Samples_test = test
    print(X_train.shape, X_test.shape,(X_test.shape[0]/2, X_test.shape[1]))
    ############ feature normalization #########
    # scaler = StandardScaler()
    # X_train_len = X_train.shape[0]
    # X = np.row_stack([X_train, X_test])
    # X = scaler.fit_transform(X)
    # X_train = X[:X_train_len]
    # X_test = X[X_train_len:]
    ############ feature selection #########
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
    val_samples, test_samples = train_test_split(df, test_size=0.5, random_state=0, stratify=df['YR'])

    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    X_val = val_samples.values
    val_data = (X_val, Y_val)

    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_test = test_samples.values
    test_data = (X_test, Y_test)

    # dataset = [train_data, val_data, test_data]
    print('start training')
    model = build_model(n_in=X_train.shape[1],
             learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
             lr_decay=lr_decay, momentum=momentum,
             L2_reg=L2_reg, L1_reg=L1_reg,
             activation="relu",
             dropout=dropout,
             input_dropout=None)
    # model = train_and_test(model, train_data, val_data=val_data, n_epochs=n_epochs, batch_size=batch_size)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=1)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test=='EA'], x_test_scr[R_test=='EA']
    Y_EAA, scr_EAA = Y_test[R_test == 'EAA'], x_test_scr[R_test == 'EAA']
    EA_AUC, EAA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_EAA, scr_EAA)
    print(A_AUC, EA_AUC, EAA_AUC)
    return


def transfer_learning(train, test, seed=0, batch_size=32, k=-1,
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

    df = pd.DataFrame(X_test)
    df['Y']= Y_test
    df['R']= R_test
    df['YR']=df['Y'].map(str) + df['R']
    val_samples, test_samples = train_test_split(df, test_size=0.5, random_state=seed, shuffle=True, stratify=df['YR'])
    Y_val, R_val = val_samples['Y'].values, val_samples['R'].values
    Y_test, R_test = test_samples['Y'].values, test_samples['R'].values
    val_samples = val_samples.drop(columns=['Y', 'R', 'YR'])
    test_samples = test_samples.drop(columns=['Y', 'R', 'YR'])
    X_val, X_test = val_samples.values, test_samples.values

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test)

    EA_X_train, EA_Y_train = X_train[R_train=='EA'], Y_train[R_train=='EA']
    EA_X_val, EA_Y_val = X_val[R_val == 'EA'], Y_val[R_val == 'EA']
    EA_X_test, EA_Y_test = X_test[R_test == 'EA'], Y_test[R_test == 'EA']

    EAA_X_train, EAA_Y_train = X_train[R_train == 'EAA'], Y_train[R_train == 'EAA']
    EAA_X_val, EAA_Y_val = X_val[R_val == 'EAA'], Y_val[R_val == 'EAA']
    EAA_X_test, EAA_Y_test = X_test[R_test == 'EAA'], Y_test[R_test == 'EAA']

    EA_train_data = (EA_X_train, EA_Y_train)
    EA_val_data = (EA_X_val, EA_Y_val)
    EA_test_data = (EA_X_test, EA_Y_test)

    EAA_train_data = (EAA_X_train, EAA_Y_train)
    EAA_val_data = (EAA_X_val, EAA_Y_val)
    EAA_test_data = (EAA_X_test, EAA_Y_test)

    Map = {}
    print('start training mixture learning')
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout, input_dropout=None)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=val_data, batch_size=batch_size, epochs=n_epochs, callbacks=[es], verbose=1)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EA'], x_test_scr[R_test == 'EA']
    Y_AA, scr_AA = Y_test[R_test == 'EAA'], x_test_scr[R_test == 'EAA']
    EA_AUC, EAA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_AA, scr_AA)
    Map["Mix0"], Map["Mix1"], Map["Mix2"] = A_AUC, EA_AUC, EAA_AUC

    print('start training independent EA')
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout, input_dropout=None)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs, callbacks=[es], verbose=1)
    EA_test_scr = np.round(model.predict(EA_X_test), decimals=3)
    EA_AUC = roc_auc_score(EA_Y_test, EA_test_scr)
    Map['Ind1'] = EA_AUC

    print('start training independent EAA')
    model = build_model(n_in=X_train.shape[1], learning_rate=0.01, hidden_layers_sizes=hiddenLayers,
                        lr_decay=lr_decay, momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
                        input_dropout=None)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=200)
    model.fit(EAA_train_data[0], EAA_train_data[1], validation_data=EAA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=1)
    EAA_test_scr = np.round(model.predict(EAA_X_test), decimals=3)
    EAA_AUC = roc_auc_score(EAA_Y_test, EAA_test_scr)
    Map['Ind2'] = EAA_AUC

    print('start training naive transfer')
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
                        lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
                        input_dropout=None)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=1)
    EAA_test_scr = np.round(model.predict(EAA_X_test), decimals=3)
    Naive_AUC = roc_auc_score(EAA_Y_test, EAA_test_scr)
    Map['Naive_Transfer'] = Naive_AUC

    print('start training transfer learning')
    model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
                        lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
                        input_dropout=None)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=EA_val_data, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=1)
    K.set_value(model.optimizer.lr, 0.15)
    for i in range(n_epochs):
        model.train_on_batch(x=EAA_train_data[0], y=EAA_train_data[1])
    prob = np.round(model.predict(EAA_X_test), decimals=3)
    A_AUC = roc_auc_score(EAA_Y_test, prob)
    Map['TL2'] = A_AUC
    # Map['TL2'] = dann_transfer(train, test, seed=seed, k=500, batch_size=32, lr=0.01, momentum=0.9, nepoch=100)
    # print('start training transfer learning')
    # model = build_model(n_in=X_train.shape[1], learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,
    #                     lr_decay=lr_decay,
    #                     momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout,
    #                     input_dropout=None)
    # es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=200)
    # model.fit(EAA_train_data[0], EAA_train_data[1], validation_data=EAA_val_data, batch_size=batch_size, epochs=n_epochs,
    #           callbacks=[es], verbose=1)
    # K.set_value(model.optimizer.lr, 0.15)
    # for i in range(n_epochs):
    #     model.train_on_batch(x=EA_train_data[0], y=EA_train_data[1])
    # prob = np.round(model.predict(EA_X_test), decimals=3)
    # A_AUC = roc_auc_score(EA_Y_test, prob)
    # Map['TL1'] = A_AUC

    df = pd.DataFrame(Map, index=[seed])
    df = df[['Mix0', 'Mix1', 'Mix2', 'Ind1', 'Ind2', 'Naive_Transfer', 'TL2']]
    df.to_csv('Lung_EAA_Result/' + str(seed) + '.csv')
    print(df)
    return df


def super_transfer(train, test, seed=0, batch_size=32, k=-1, tune_lr=0.15, tune_decay=0.003, tune_batch=20,
               learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9, n_epochs_tune=100,
               L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    Agg, EA, EAA = prepare_data(train, test, seed=seed, k=k)
    EA_train, EA_val, EA_test = EA
    EAA_train, EAA_val, EAA_test = EAA
    EAA_X_test, EAA_Y_test = EAA_test

    # normalization
    X_train1, X_val1, X_test1 = EA_train[0], EA_val[0], EA_test[0]
    X1 = np.concatenate([X_train1, X_val1, X_test1], axis=0)
    X_train2, X_val2, X_test2 = EAA_train[0], EAA_val[0], EAA_test[0]
    X2 = np.concatenate([X_train2, X_val2, X_test2], axis=0)

    X = np.append(X1, X2, axis=0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X1 = X[:X1.shape[0]]
    X2 = X[X2.shape[0]:]

    n1, n2, n3 = X_train1.shape[0], X_val1.shape[0], X_test1.shape[0]
    X_train1, X_val1, X_test1 = X1[:n1], X1[n1:n1+n2], X1[n1+n2:]
    n1, n2, n3 = X_train2.shape[0], X_val2.shape[0], X_test2.shape[0]
    X_train2, X_val2, X_test2 = X2[:n1], X2[n1:n1 + n2], X2[n1 + n2:]

    EA_train[0], EA_val[0], EA_test[0] = X_train1, X_val1, X_test1
    EAA_train[0], EAA_val[0], EAA_test[0] = X_train2, X_val2, X_test2
    #### end of the normalization ########################################

    n_in = EA_train[0].shape[1]
    model = build_model(n_in=n_in, learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(EA_train[0], EA_train[1], validation_data=EA_val, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    K.set_value(model.optimizer.lr, tune_lr)
    K.set_value(model.optimizer.decay, tune_decay)
    model.fit(EAA_train[0], EAA_train[1], validation_data=EAA_val, batch_size=tune_batch, epochs=n_epochs_tune,
              callbacks=[es], verbose=0)
    # for i in range(n_epochs):
    #     model.train_on_batch(x=EAA_train[0], y=EAA_train[1])
    #     prob = np.round(model.predict(EAA_X_test), decimals=3)
    #     print(i, roc_auc_score(EAA_Y_test, prob))

    prob = np.round(model.predict(EAA_X_test), decimals=3)
    A_AUC = roc_auc_score(EAA_Y_test, prob)
    return A_AUC

def mixture_learning(Agg, EA, EAA, seed=0, batch_size=32, k=-1,
               learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9,
               L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    # Agg, EA, EAA = prepare_data(train, test, seed=seed, k=k, minor='EAA')
    Aggr_train, Aggr_val, Aggr_test = Agg
    X_test, Y_test, R_test = Aggr_test
    n_in = Aggr_train[0].shape[1]
    model = build_model(n_in=n_in, learning_rate=learning_rate, hidden_layers_sizes=hiddenLayers,lr_decay=lr_decay,
                        momentum=momentum, L2_reg=L2_reg, L1_reg=L1_reg, activation="relu", dropout=dropout)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=0, patience=200)
    model.fit(Aggr_train[0], Aggr_train[1], validation_data=Aggr_val, batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)

    x_test_scr = np.round(model.predict(X_test), decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EA'], x_test_scr[R_test == 'EA']
    Y_AA, scr_AA = Y_test[R_test == 'EAA'], x_test_scr[R_test == 'EAA']
    EA_AUC, EAA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_AA, scr_AA)
    Map = {}
    Map["Aggr"] = A_AUC
    Map["EA_AUC"] = EA_AUC
    Map["EAA_AUC"] = EAA_AUC
    return (seed, Map)

def independent_learning(data, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = build_model(n_in=train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=200)
    model.fit(train_data[0], train_data[1], validation_data=(val_data[0], val_data[1]), batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    x_test_scr = np.round(model.predict(X_test), decimals=3)
    return roc_auc_score(Y_test, x_test_scr)

def naive_transfer(EA, Minor, **kwds):
    batch_size = kwds.pop('batch_size')
    n_epochs = kwds.pop('n_epochs')
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test

    model = build_model(n_in=EA_train_data[0].shape[1], **kwds)
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=200)
    model.fit(EA_train_data[0], EA_train_data[1], validation_data=(EA_val_data[0], EA_val_data[1]), batch_size=batch_size, epochs=n_epochs,
              callbacks=[es], verbose=0)
    Minor_test_scr = np.round(model.predict(Minor_test_X), decimals=3)
    Naive_AUC = roc_auc_score(Minor_test_Y, Minor_test_scr)
    return Naive_AUC


def get_data(file):
    A = loadmat(file)
    R = A['R'][0]
    Y = A['Y'][0].astype('int32')
    X = A['X'].astype('float32')
    Samples = A['Samples'][0]
    # print(Samples)
    # Samples = [s.strip() for s in Samples]
    R1 = [str(row[0]) for row in R]
    data = (X, Y, R1, Samples)
    return data


def run_cv(seed):

    train = get_data('0_train_X3.mat')
    test = get_data('0_test_X3.mat')

    parametrs_mix = {'k': 1000, 'batch_size': 100, 'momentum': 0.9, 'n_epochs': 100,
                     'learning_rate': 0.15, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}
    parametrs_mix = {'k': 500, 'batch_size': 100, 'momentum': 0.9, 'n_epochs': 200,
                     'learning_rate': 0.15, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}
    parametrs_mix = {'k': 700, 'batch_size': 32, 'momentum': 0.9, 'n_epochs': 200,
                     'learning_rate': 0.15, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [200]}
    parametrs_mix = {'k': 200, 'batch_size': 32, 'momentum': 0.9, 'n_epochs': 200,
                     'learning_rate': 0.15, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}
    parametrs_mix = {'k': 1000, 'batch_size': 32, 'momentum': 0.9, 'n_epochs': 200, 'seed':seed,
                     'learning_rate': 0.15, 'lr_decay': 0.005, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}


    n_epoches = 200
    k=1000
    para_mix = {'k': k, 'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoches,
                     'learning_rate': 0.15, 'lr_decay': 0.003, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}
    para_ind_EA = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoches,
                     'learning_rate': 0.15, 'lr_decay': 0.003, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_ind_EAA = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoches,
                     'learning_rate': 0.15, 'lr_decay': 0.003, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_naive = {'batch_size': 32, 'momentum': 0.9, 'n_epochs': n_epoches,
                     'learning_rate': 0.25, 'lr_decay': 0.003, 'dropout': 0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hidden_layers_sizes': [100]}
    para_super = {'batch_size': 50, 'momentum': 0.9, 'n_epochs': 250, 'n_epochs_tune': 50, 'k': 500, 'tune_batch': 30,
                  'learning_rate': 0.25, 'lr_decay': 0.003, 'dropout': 0.5, 'tune_lr': 0.15, 'tune_decay': 0.003,
                  'seed':seed, 'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}

    for lr in [0.25, 0.2, 0.15, 0.1, 0.05, 0.01]:
        for decay in [0.003, 0]:
            for k in [500, 300, 200]:
                para_super['learning_rate'], para_super['lr_decay'], para_super['k'] = lr, decay, k
                tl = super_transfer(train, test, **para_super)
                print(lr, decay, k, tl)

    Agg, EA, EAA = prepare_data(train, test, seed=seed, k=k, minor='EAA')
    Map = {}
    res = pd.DataFrame()
    for seed in range(20):
        seed, Map = mixture_learning(Agg, EA, EAA, **para_mix)
        Map['ind_1'] = independent_learning(EA, **para_ind_EA)
        Map['ind_2'] = independent_learning(EAA, **para_ind_EAA)
        Map['naive'] = naive_transfer(EA, EAA, **para_naive)
        Map['tl2'] = super_transfer(train, test, **para_super)
        df = pd.DataFrame(Map, index=[seed])
        res = res.append(df)
        print(df)

    return res

if __name__ == '__main__':

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    arguments = sys.argv
    print(arguments)
    seed = int(arguments[1])
    df = run_cv(seed)
    df.to_csv("Result_Lung/{}.csv".format(seed))
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

