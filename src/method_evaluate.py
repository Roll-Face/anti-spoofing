import numpy as np
import torch.nn.functional as F
from sklearn.metrics import roc_curve


def get_tp_fp_rates(y_true, y_pred):
    fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    return fpr, tpr, threshold


def get_far_frr(tp, tn, fp, fn):
    far = fp / (fp + tn)
    frr = fn / (tp + fn)
    return far, frr


def get_equal_error_rate(tpr, fpr):
    frr = 1 - np.asarray(fpr)  # False Rejection Rate
    min_diff = 1
    i_min = 0
    j_min = 0

    for i in range(0, len(frr)):
        for j in range(0, len(fpr)):
            diff = np.absolute(frr[i] - fpr[j])
            if diff < min_diff and diff > 0.2:
                min_diff = diff
                i_min = i
                j_min = j
    return np.min([frr[i_min], fpr[j_min]])
