import pandas as pd
import numpy as np
from scipy.io import savemat
from scipy.stats import truncnorm

pd.set_option('max_colwidth', 400)

def generateY_h(minor='AFR', rho=0.8, h2=0.5):

    x_1, x_2 = np.random.uniform(-1, 1, (10000, 500)), np.random.uniform(-1, 1, (2000, 500))
    df1, df2 = pd.DataFrame(x_1), pd.DataFrame(x_2)

    N = df1.shape[1]
    x1 = np.random.uniform(low=-1, high=1.0, size=N)
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
    thr = df1['scr'].median()
    df1['Y'] = df1['scr'] >= thr
    df1["Y"] = df1["Y"].astype(int)

    df2['R'] = minor
    thr = df2['scr'].median()
    df2['Y'] = df2['scr'] >= thr
    df2["Y"] = df2["Y"].astype(int)

    df = df1.append(df2)
    df['YR'] = df['Y'].map(str) + df['R']
    Y, R, YR = df['Y'].values, df['R'].values, df['YR'].values
    X = df.drop(columns=['Y', 'R', 'YR', 'scr'])
    data = {'X': X.values, 'Y': Y, 'R': R, 'YR': YR}
    # savemat("Heritability/EUR_{}_{}_h{}.mat".format(minor, rho, h2), data)
    rho1 = round(np.corrcoef(x1, z1)[0][1],3)
    h21, h22 = round(np.corrcoef(scr11, scr1)[0][1] ** 2, 3), round(np.corrcoef(scr22, scr2)[0][1] ** 2, 3)
    print("For EUR-{} rho={}, h2={}".format(minor, rho, h2), 'The real rho and h2 are: ', rho1, h21, h22)


generateY_h('AFR', 0.54, h2=0.5)
generateY_h('AMR', 0.80, h2=0.5)
generateY_h('EAS', 0.58, h2=0.5)
generateY_h('SAS', 0.77, h2=0.5)
#
generateY_h('AFR', 0.54, h2=0.25)
generateY_h('AMR', 0.80, h2=0.25)
generateY_h('EAS', 0.58, h2=0.25)
generateY_h('SAS', 0.77, h2=0.25)


