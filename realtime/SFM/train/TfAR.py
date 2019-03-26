from __future__ import print_function
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.contrib.timeseries.python.timeseries import  NumpyReader

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

def main(_):
    data = np.zeros((3, 1680), dtype=np.float32)
    all_data = np.load('../dataset/data.npy')
    pre = np.zeros((3, 168), dtype=np.float32)
    print(os.listdir('../dataset/'))
    anomaly = np.load('../dataset/anomaly_all.npy')


    val_split = round(0.9 * all_data.shape[1])
    test_split = round(0.1 * all_data.shape[1])
    X_train = all_data[:, :val_split]
    X_test = all_data[:, -test_split:]
    X_test_label = anomaly[:, -test_split:].flatten()
    print(val_split,test_split, X_test_label.shape)

    x = np.array(range(val_split))
    plt.figure(figsize=(16, 8))
    ts = ['TS6', 'TS16', 'TS34']
    for i in range(3):
        x_train = X_train[i]
        data = {
            tf.contrib.timeseries.TrainEvalFeatures.TIMES: x,
            tf.contrib.timeseries.TrainEvalFeatures.VALUES: x_train,
        }

        reader = NumpyReader(data)
        train_input_fn = tf.contrib.timeseries.RandomWindowInputFn(
            reader, batch_size=16, window_size=40)

        ar = tf.contrib.timeseries.ARRegressor(
            periodicities=24, input_window_size=30, output_window_size=10,
            num_features=1,
            loss=tf.contrib.timeseries.ARModel.NORMAL_LIKELIHOOD_LOSS)

        ar.train(input_fn=train_input_fn, steps=6000)

        evaluation_input_fn = tf.contrib.timeseries.WholeDatasetInputFn(reader)
        # keys of evaluation: ['covariance', 'loss', 'mean', 'observed', 'start_tuple', 'times', 'global_step']
        evaluation = ar.evaluate(input_fn=evaluation_input_fn, steps=1)

        (predictions,) = tuple(ar.predict(
            input_fn=tf.contrib.timeseries.predict_continuation_input_fn(
                evaluation, steps=test_split)))
        pre[i] = predictions['mean'].reshape(-1)
        ax = plt.subplot(311+i)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.plot(data['times'].reshape(-1), data['values'].reshape(-1), color='orangered', lw = 2 , label='Original Data')
        #plt.plot(evaluation['times'].reshape(-1), evaluation['mean'].reshape(-1),  linewidth=1.0, linestyle=':', color = 'sandybrown',marker='*',label='evaluation')
        plt.plot(predictions['times'].reshape(-1), predictions['mean'].reshape(-1), color='orangered', linestyle = '--' ,lw = 2 ,label='Prediction Data' )
        if i == 2 :
            ax.set_xlabel('TIMESTEMPS', fontsize=20)
        ax.set_ylabel(ts[i], fontsize=20)
        ax.legend(loc = 2, fontsize=10)

    plt.savefig('ar.pdf')
    plt.show()
    """
    label = np.load('../dataset/anomally_each.npy')
    labels = label[:, -test_split:]
    diff = pre - X_test
    tpr, fpr, auc = caulcute_tf(diff, labels)
    roc = np.zeros((2, len(tpr)), dtype=np.float32)
    roc[0] = tpr
    roc[1] = fpr
    np.save('./Roc/AR_ROC', roc)
    """

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()