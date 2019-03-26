import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
import distutils.util


def calculate_all(data,label,threshold):
    tpr_c = np.zeros(len(threshold), dtype=np.float32)
    fpr_c = np.zeros(len(threshold), dtype=np.float32)
    acu_c = np.zeros(len(threshold), dtype=np.float32)

    real_label = label[0] + label[1] + label[2]
    index = 0
    for i in range(len(threshold)):
        th = threshold[i];
        label_0 = [1 if j > th[0] else 0 for j in data[0]]
        label_1 = [1 if k > th[1] else 0 for k in data[1]]
        label_2 = [1 if m > th[2] else 0 for m in data[2]]
        pre_label = label_0 + label_1 + label_2
        pre_label = [0 if n == 0 else 1 for n in pre_label]
        tp = 0
        fn = 0
        fp = 0
        tn = 0
        for i_index in range(len(real_label)) :
            ii = real_label[i_index]
            jj = pre_label[i_index]
            if jj == 0 and ii == 0:
                tp += 1
            elif ii == 1 and jj == 1:
                tn += 1
            elif ii == 0 and jj == 1:
                fn += 1
            else:
                fp += 1
        tpr_i = tp / (tp + fn)
        fpr_i = fp / (fp + tn)
        acu_i = (tp + tn) / (tp + tn + fp + fn)
        tpr_c[index] = tpr_i
        fpr_c[index] = fpr_i
        acu_c[index] = acu_i
        index += 1

    auc_c = 0
    for i_t in range(len(tpr_c) - 1):
        auc_c = auc_c + (fpr_c[i_t+1] - fpr_c[i_t]) * (tpr_c[i_t+1] + tpr_c[i_t])
    auc_c = auc_c / 2
    print(np.max(acu_c))
    return tpr_c, fpr_c, auc_c




def caulcute_tf(diff,labels):
    abs_diff = np.abs(diff)
    sorted_diff = np.sort(abs_diff,axis=1)
    threshold = np.zeros((len(diff[0])+1, 3), dtype=np.float32)
    threshold[0] = np.array([0,0,0])
    for i in range(len(sorted_diff[0])):
        th = [sorted_diff[0][i],sorted_diff[1][i],sorted_diff[2][i]]
        threshold[i+1] = np.array(th)
    tpr,fpr,auc = calculate_all(abs_diff,labels,threshold)
    return tpr,fpr,auc



if __name__ == '__main__':
    data = pd.read_csv('all_diff_data.csv')
    diff = np.zeros((3, 167), dtype=np.float32)
    anomaly = np.zeros((3, 167), dtype=np.float32)
    tit = ['diff_ts6','diff_ts16','diff_ts34']
    a_tit = ['anomaly_ts6','anomaly_ts16','anomaly_ts34']

    for i in range(3):
        var = [tit[i]]
        data_var = np.array(data[var])
        diff[i] = np.transpose(data_var)
        anomaly_i = [a_tit[i]]
        anomaly_i = np.array(data[anomaly_i])
        anomaly[i] = np.transpose(anomaly_i)

    tpr,fpr,auc = caulcute_tf(diff, anomaly)
    roc = np.zeros((2, len(tpr)), dtype=np.float32)
    roc[0] = tpr
    roc[1] = fpr
    df = pd.DataFrame(roc.transpose())
    df.to_csv('SFM_roc.csv')

    """
    plt.figure()
    plt.title("ROC curve of %s (AUC = %.4f)" % ('SFM', auc))
    plt.plot(fpr,tpr)
    plt.show()
    """

