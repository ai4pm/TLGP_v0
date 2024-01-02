import torch
import torch.optim as optim
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
from sklearn.linear_model import LogisticRegression

DEVICE = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')

class TL_PRS(nn.Module):
    def __init__(self, device, n_input, beta_pre, gama_pre, n_cov):
        super(TL_PRS, self).__init__()
        self.device = device
        self.n_cov = 0
        self.tau = nn.Linear(n_input, 1)
        self.beta_pre = nn.Linear(n_input, 1)
        self.beta_pre.weight = beta_pre

        if n_cov > 0:
            self.n_cov = n_cov
            self.gamma_pre = nn.Linear(n_cov, 1)
            self.gamma_pre.weight = gama_pre
            self.delta = nn.Linear(n_cov, 1)

    def forward(self, G, C):
        loss = self.beta_pre(G) + self.tau(G)
        if self.n_cov > 0: loss = loss + self.gamma_pre(C) + self.delta(C)
        return loss

# k # of features
def TL_PRS_scr(seed, X_s, X_t, Y_s, Y_t, k, X_test, lr=0.01, nepoch=100, batch=30):
    torch.random.manual_seed(seed)
    n_cov = X_t.shape[1] - k

    log = LogisticRegression(solver='lbfgs')
    log.fit(X_s, Y_s)
    weights = log.coef_
    beta_pre = torch.from_numpy(weights[:,:k])
    gama_pre = torch.from_numpy(weights[:,k:])
    beta_pre, gama_pre = beta_pre.type(torch.FloatTensor), gama_pre.type(torch.FloatTensor)

    beta_pre = Parameter(beta_pre)
    gama_pre = Parameter(gama_pre)

    C_t = X_t[:,k:]
    X_t = X_t[:,:k]

    model = TL_PRS(DEVICE, k, beta_pre, gama_pre, n_cov).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0.001)

    epson = 1e-4
    best_err = float('inf')
    n_batch = X_t.shape[0] // batch
    loss_fn = nn.MSELoss()
    for epoch in range(nepoch):

        model.train()
        err = 0
        for i in range(n_batch):
            s, e = i*batch, (i+1) * batch
            G, Y, C = X_t[s:e], Y_t[s:e], C_t[s:e]
            G, Y, C = torch.from_numpy(G), torch.from_numpy(Y), torch.from_numpy(C)

            G, Y, C = G.type(torch.FloatTensor), Y.type(torch.FloatTensor), C.type(torch.FloatTensor)
            G, Y, C = G.to(DEVICE), Y.to(DEVICE), C.to(DEVICE)
            scr = model(G, C)

            noise = torch.tensor(np.random.normal(0, 1, size=(batch,1)), dtype=torch.float)
            scr += noise
            scr = torch.squeeze(scr)
            err += loss_fn(scr, Y)

        err /= n_batch
        # print(epoch, err)
        if abs(best_err - err) < epson: break
        if err < best_err: best_err = err.data
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    model.eval()
    C_test = X_test[:,k:]
    X_test = X_test[:,:k]
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor).to(DEVICE)
    C_test = torch.from_numpy(C_test).type(torch.FloatTensor).to(DEVICE)
    scr = model(X_test, C_test)

    return scr.data


def partial_hazard(risk, e):
    eps = 1e-7
    if e.dtype is torch.bool: e = e.float()
    events = e.view(-1)
    risk = risk.view(-1)
    gamma = risk.max()
    log_cumsum_h = risk.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - risk.sub(log_cumsum_h).mul(events).sum().div(events.sum())

def run_CPH_DL_ROI(seed, df_train, df_att, df_test, L2=0.0001, lr=0.01, momentum=0.9, gamma=0.2, nepoch=100):
    T, E, R = df_train['T'].values.astype('int32'), df_train['E'].values.astype('int32'), df_train['Gender'].values
    X = df_train.drop(columns=['T', 'E', 'Gender']).values.astype('float32')
    T_att, E_att, R_att = df_att['T'].values.astype('int32'), df_att['E'].values.astype('int32'), df_att['Gender'].values
    X_att = df_att.drop(columns=['T', 'E', 'Gender']).values.astype('float32')

    X, T, E, R = prepare_data([X, T, E, R])
    X_att, T_att, E_att, R_att = prepare_data([X_att, T_att, E_att, R_att])
    loss_cox = partial_hazard

    in_dim = X.shape[1]
    torch.random.manual_seed(seed)
    model = CPH_DL_ROI(DEVICE, in_dim).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), weight_decay=L2, lr=lr, momentum=momentum)

    epson = 1e-4
    best_err = float('inf')
    for epoch in range(nepoch):
        model.train()
        x_tar, e_tar = torch.from_numpy(X), torch.from_numpy(E)
        x_att, e_att = torch.from_numpy(X_att), torch.from_numpy(E_att)

        e_tar, e_att = e_tar.type(torch.LongTensor), e_att.type(torch.LongTensor)
        x_tar, e_tar = x_tar.to(DEVICE), e_tar.to(DEVICE)
        x_att, e_att = x_att.to(DEVICE), e_att.to(DEVICE)
        main_output, attentioner_output = model(tar_fature=x_tar, att_feature=x_att)
        err_main = loss_cox(main_output, e_tar)
        err_att = loss_cox(attentioner_output, e_att)

        err = (1 - gamma) * err_main + gamma * (err_att)
        if abs(best_err - err) < epson: break
        if err < best_err: best_err = err.data
        optimizer.zero_grad()
        err.backward()
        optimizer.step()

    model.eval()
    X_test = df_test.values.astype('float32')
    X_test = torch.from_numpy(X_test)
    X_test = X_test.to(DEVICE)
    scr = model(tar_fature=X_test, att_feature=None, istesting=True)
    scr = torch.flatten(-scr)

    weights = model.main.cox_regression.c_f2.weight
    return scr.data, weights.data.numpy().flatten()










