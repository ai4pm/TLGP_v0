from numpy.random import seed
import pandas as pd
import random as rn
import os,sys
import timeit
import numpy as np
from datetime import datetime

from scipy.io import loadmat
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import SelectKBest

sys.path.append('Logistic/')
from TL_PRS import TL_PRS_scr



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

seed(11111)
os.environ['PYTHONHASHSEED'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"
rn.seed(11111)


path = 'Lung/'


def prepare_data(train, test, seed=0, k=1000, minor='EAA'):
    X_train, Y_train, R_train, Samples_train = train
    X_test, Y_test, R_test, Samples_test = test
    # print(X_train.shape, X_test.shape, (X_test.shape[0] / 2, X_test.shape[1]))
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

def prepare_data1(train, test, seed=0, k=1000, minor='EAA'):
    X_train, Y_train, R_train, Samples_train = train
    X_test, Y_test, R_test, Samples_test = test

    R_train = np.asarray(R_train)
    X_train_t, Y_train_t = X_train[R_train == minor], Y_train[R_train == minor]

    select = SelectKBest(k=k)
    select.fit(X_train_t, Y_train_t)
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
    f_path = path + 'Covar/'
    df1 = pd.read_csv(f_path + 'c1_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df2 = pd.read_csv(f_path + 'c3_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                   usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df3 = pd.read_csv(f_path + 'c6_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df4 = pd.read_csv(f_path + 'c8_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df5 = pd.read_csv(f_path + 'c10_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])
    df6 = pd.read_csv(f_path + 'c14_Lung_Cancer_Subject_Phenotypes.txt', sep='\t', skiprows=10, index_col='subject_ID',
                      usecols=['subject_ID', 'age_category', 'sex', 'packyear_category', 'typesmok'])

    df = pd.concat([df1, df2, df3, df4, df5, df6])
    df_map = pd.read_csv(f_path + 'c1.csv', sep='\t', skiprows=10, index_col='subject_ID', usecols=['subject_ID', 'sample_ID'])
    df = df.join(df_map, how='left')
    df = df.reset_index().set_index('sample_ID')
    df = df.drop(columns=['subject_ID'])

    df_sex_dummy = pd.get_dummies(df['sex'], prefix='sex', drop_first=True)
    df_typesmk_dummy = pd.get_dummies(df['typesmok'], prefix='typesmk', drop_first=True)
    df = df.drop(columns=['sex', 'typesmok'])
    df1 = df.join(df_sex_dummy, how='left')
    df1 = df1.join(df_typesmk_dummy, how='left')

    df_data = pd.DataFrame(X, index=samples)
    df_data['Y'] = Y
    df_data['R'] = R
    df_data = df_data.join(df1, how='inner')
    Y, R = df_data['Y'].values, df_data['R'].values
    df_data = df_data.drop(columns=['Y', 'R'])
    X = df_data.values

    df1 = pd.DataFrame()
    df1['Y'], df1['R'] = Y,R
    df1['YR'] = df1['Y'].astype(str) + df1['R']
    return X, Y, R

def mixture_learning(Agg, minor='EAA'):
    Aggr_train, Aggr_val, Aggr_test = Agg
    X_test, Y_test, R_test = Aggr_test
    model = LogisticRegression(random_state=0, penalty='l2', solver='sag')
    model.fit(Aggr_train[0], Aggr_train[1])

    x_test_scr = np.round(model.predict_proba(X_test)[:,1], decimals=3)
    A_AUC = roc_auc_score(Y_test, x_test_scr)
    Y_EA, scr_EA = Y_test[R_test == 'EA'], x_test_scr[R_test == 'EA']
    Y_AA, scr_AA = Y_test[R_test == minor], x_test_scr[R_test == minor]
    EA_AUC, EAA_AUC = roc_auc_score(Y_EA, scr_EA), roc_auc_score(Y_AA, scr_AA)
    return A_AUC, EA_AUC, EAA_AUC

def independent_learning(data):
    train_data, val_data, test_data = data
    X_test, Y_test = test_data
    model = LogisticRegression(random_state=0, penalty='l2', solver='sag')
    model.fit(train_data[0], train_data[1])
    x_test_scr = np.round(model.predict_proba(X_test)[:,1], decimals=3)
    return roc_auc_score(Y_test, x_test_scr)

def naive_transfer(EA, Minor):
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    Minor_test_X, Minor_test_Y = Minor_test

    model = LogisticRegression(max_iter=200)
    model.fit(EA_val_data[0], EA_val_data[1])
    Minor_test_scr = np.round(model.predict_proba(Minor_test_X)[:,1], decimals=3)
    Naive_AUC = roc_auc_score(Minor_test_Y, Minor_test_scr)
    return Naive_AUC


def transfer(EA, Minor, k=500, batch=50):
    EA_train_data, EA_val_data, EA_test_data = EA
    Minor_train_data, Minor_val_data, Minor_test = Minor
    X_test, Y_test = Minor_test

    X_s, Y_s = EA_train_data
    X_t, Y_t = Minor_train_data

    scr = TL_PRS_scr(0, X_s, X_t, Y_s, Y_t, k, X_test, lr=0.001, batch=batch)
    auc = roc_auc_score(Y_test, scr)
    return auc

def get_data(file):
    A = loadmat(file)
    R = A['R'][0]
    Y = A['Y'][0].astype('int32')
    X = A['X'].astype('float32')
    Samples = A['Samples'][0]
    R1 = [str(row[0]) for row in R]
    data = (X, Y, R1, Samples)
    return data

def run_cv(k=500):

    train = get_data(path + '0_train_X3.mat')
    test = get_data(path + '0_test_X3.mat')

    res = []
    for seed in range(20):
        Agg, EA, EAA = prepare_data(train, test, seed=seed, k=k, minor='EAA')
        mix0, mix1, mix2 = mixture_learning(Agg)
        ind1 = independent_learning(EA)
        ind2 = independent_learning(EAA)
        naive = naive_transfer(EA, EAA)
        tl2 = transfer(EA, EAA, k=k)
        row = [mix0, mix1, mix2, ind1, ind2, naive, tl2]
        res.append(row)
        print(seed, row)
    df = pd.DataFrame(res)
    return df

if __name__ == '__main__':

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Start Time =", current_time)
    arguments = sys.argv
    print(arguments)
    df = run_cv(k=500)
    print(df)
    # df = run_cv(k=1000)
    # print(df)
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("End Time =", current_time)

