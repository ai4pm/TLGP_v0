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
    # loss_fn = nn.MSELoss()
    loss_fn = nn.BCELoss()
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
            scr = torch.sigmoid(scr)
            scr = torch.squeeze(scr)
            err += loss_fn(scr, Y)

        err /= n_batch
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

