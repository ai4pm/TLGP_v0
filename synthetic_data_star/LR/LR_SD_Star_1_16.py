from numpy.random import seed
import pandas as pd
import random as rn
import os,sys
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from openpyxl import load_workbook

from TL_PRS import TL_PRS_scr
from get_performance import get_performance


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 1000)

seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)


def make_df(Y_test, R, x_test_scr):
    df = pd.DataFrame()
    df['Y'] = Y_test
    df['R'] = R
    df['Scr'] = x_test_scr
    return df

def independent_learning(data, R='EA'):
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = LogisticRegression(solver='newton-cg', C=0.5)
    model.fit(train_data[0], train_data[1])

    x_test_scr = np.round(model.predict_proba(X_test), decimals=3)[:,1]
    df = make_df(Y_test, R, x_test_scr)
    return df

def mixture_learning(Aggr, minor='AA'):
    train_data, val_data, test_data = Aggr
    X_test, Y_test, R_test = test_data

    model = LogisticRegression(solver='newton-cg', C=0.5)
    model.fit(train_data[0], train_data[1])
    x_test_scr = np.round(model.predict_proba(X_test)[:,1], decimals=3)
    df = make_df(Y_test, R_test, x_test_scr)
    return df

def naive_transfer(EA, Minor):
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test
    model = LogisticRegression(solver='newton-cg', C=0.5)

    model.fit(EA_train_data[0], EA_train_data[1])
    Minor_test_scr = np.round(model.predict_proba(Minor_test_X), decimals=3)[:,1]
    df = make_df(Minor_test_Y, "", Minor_test_scr)
    return df

def transfer(EA, Minor, k=500, batch=50, lr=0.005):
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    X_test, Y_test = Minor_test

    X_s, Y_s = EA_train_data
    X_t, Y_t = Minor_train_data

    scr = TL_PRS_scr(0, X_s, X_t, Y_s, Y_t, k, X_test, batch=batch, lr=lr)
    df = make_df(Y_test, "", scr.numpy())
    return df

def get_data(file, seed=0, minor='AMR'):
    A = loadmat(file)
    R = A['R'][0]
    R = [row[0] for row in R]
    Y = A['Y'][0].astype('int32')
    X = A['X'].astype('float32')

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    df = pd.DataFrame(X)
    print(len(R), len(Y))
    df['R'], df['Y'] = R, Y
    df['YR'] = df['Y'].map(str) + df['R'].map(str)

    train, test = train_test_split(df, test_size=0.2, random_state=11, shuffle=True, stratify=df['YR'])
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

def run_sd_star(SD, metric='auc'):

    path = '../../data/SD_data_Star/'
    print(path + Map_SD[SD], SD)
    res = []
    for seed in range(20):

        Aggr, EA, Minor = get_data(path + Map_SD[SD], seed=seed, minor=Map_Minor[SD-1])
        df_mix0 = mixture_learning(Aggr, minor=Map_Minor[SD-1])
        df_mix1 = df_mix0[df_mix0['R'] == 'EUR']
        df_mix2 = df_mix0[df_mix0['R'] == Map_Minor[SD-1]]
        df_ind_1 = independent_learning(EA, R='EA')
        df_ind_2 = independent_learning(Minor, R=Map_Minor[SD-1])
        df_naive = naive_transfer(EA, Minor)
        df_tl_2 = transfer(EA, Minor, k=500, batch=50, lr=0.005)

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
        df = run_sd_star(SD)


if __name__ == '__main__':
    main()
