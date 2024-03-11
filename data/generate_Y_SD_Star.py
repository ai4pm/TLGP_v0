import pandas as pd
import numpy as np
from scipy.io import savemat, loadmat
from scipy.stats import truncnorm

pd.set_option('max_colwidth', 400)

def generateY_h(minor='AFR', rho=0.8, h2=0.5):

    f_path = "../../data/SD_data/EUR_{}_{}_h{}.mat".format(minor, rho, h2)
    A = loadmat(f_path)
    R = A['R']
    X = A['X'].astype('float32')
    df = pd.DataFrame(X)
    df['R'] = R
    df1, df2 = df[df['R']=='EUR'], df[df['R']==minor]
    df1 = df1.drop(columns='R')
    df2 = df2.drop(columns='R')
    print(f_path, df1.shape, df2.shape)

    N = df1.shape[1]
    x1 = truncnorm.rvs(-1, 1, size=N)
    x1 = x1 / np.linalg.norm(x1)
    x2 = truncnorm.rvs(-1, 1, size=N)
    x2 = x2 / np.linalg.norm(x2)

    z1 = rho * x1 + np.sqrt(1 - rho**2) * x2
    z1 = z1 / np.linalg.norm(z1)

    scr1 = df1.__matmul__(x1)
    scr2 = df2.__matmul__(z1)

    scr1 = scr1 / np.linalg.norm(scr1)
    x2 = truncnorm.rvs(-1, 1, size=df1.shape[0])
    x2 = x2 / np.linalg.norm(x2)
    scr11 = np.sqrt(h2) * scr1 + np.sqrt(1 - h2) * x2
    scr11 = scr11 / np.linalg.norm(scr11)

    scr2 = scr2 / np.linalg.norm(scr2)
    x2 = truncnorm.rvs(-1, 1, size=df2.shape[0])
    x2 = x2 / np.linalg.norm(x2)
    scr22 = np.sqrt(h2) * scr2 + np.sqrt(1 - h2) * x2
    scr22 = scr22 / np.linalg.norm(scr22)

    df1['scr'] = scr11
    df2['scr'] = scr22
    df1['R'] = 'EUR'
    thr = np.percentile(df1['scr'], 80)
    df1['Y'] = df1['scr'] >= thr
    df1["Y"] = df1["Y"].astype(int)

    df2['R'] = minor
    thr = np.percentile(df2['scr'], 80)
    df2['Y'] = df2['scr'] >= thr
    df2["Y"] = df2["Y"].astype(int)

    df = df1.append(df2)
    df['YR'] = df['Y'].map(str) + df['R']
    Y, R, YR = df['Y'].values, df['R'].values, df['YR'].values
    X = df.drop(columns=['Y', 'R', 'YR', 'scr'])
    data = {'X': X.values, 'Y': Y, 'R': R, 'YR': YR}
    savemat("EUR_{}_{}_h{}.mat".format(minor, rho, h2), data)
    print(df1["Y"].sum(), df2["Y"].sum())


Map_SD = { 1: "EUR_AMR_0.8_h0.5.mat", 2: "EUR_SAS_0.77_h0.5.mat", 3:"EUR_EAS_0.58_h0.5.mat", 4:"EUR_AFR_0.54_h0.5.mat",
        5:"EUR_AMR_0.8_h0.25.mat", 6:"EUR_SAS_0.77_h0.25.mat", 7:"EUR_EAS_0.58_h0.25.mat", 8:"EUR_AFR_0.54_h0.25.mat",
        10:"EUR_SAS_0.74_h0.5.mat", 11:"EUR_EAS_0.42_h0.5.mat", 12:"EUR_AFR_0.36_h0.5.mat",
        14:"EUR_SAS_0.74_h0.25.mat", 15:"EUR_EAS_0.42_h0.25.mat", 16:"EUR_AFR_0.36_h0.25.mat"
}




generateY_h('AMR', 0.8, h2=0.5)
generateY_h('SAS', 0.77, h2=0.5)
generateY_h('EAS', 0.58, h2=0.5)
generateY_h('AFR', 0.54, h2=0.5)
#
generateY_h('AMR', 0.80, h2=0.25)
generateY_h('SAS', 0.77, h2=0.25)
generateY_h('EAS', 0.58, h2=0.25)
generateY_h('AFR', 0.54, h2=0.25)

generateY_h('SAS', 0.74, h2=0.5)
generateY_h('EAS', 0.42, h2=0.5)
generateY_h('AFR', 0.36, h2=0.5)
#
generateY_h('SAS', 0.74, h2=0.25)
generateY_h('EAS', 0.42, h2=0.25)
generateY_h('AFR', 0.36, h2=0.25)



