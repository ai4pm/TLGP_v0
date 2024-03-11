
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, average_precision_score
import numpy as np
from sklearn.metrics import precision_score, confusion_matrix

def get_auc(y_true, y_pred_scr):
    return roc_auc_score(y_true, y_pred_scr)

def get_aupr(y_true, y_pred_scr):
    au_pr_score = average_precision_score(y_true, y_pred_scr)
    return au_pr_score

Map = {'auc':get_auc, 'aupr':get_aupr}

def get_performance(y_true, y_pred_scr, metric='auc'):
    return Map[metric](y_true, y_pred_scr)


