import sys


sys.path.append('C:/Users/Shaohan/PycharmProjects/ScienceProject/ScienceProject/catalog/SFM/train')
sys.path.append('C:/Users/Shaohan/PycharmProjects/ScienceProject/ScienceProject/catalog/SFM/train/science_data')
import build
import numpy as np
import argparse
import time
import pickle

np.set_printoptions(threshold=np.nan)

def predict(X_value, model_path = 'C:/Users/Shaohan/PycharmProjects/ScienceProject/ScienceProject/catalog/SFM/train/science_data/weights3660.hdf5' , hidden_dim = 50 , freq_dim = 4, learning_rate = 0.0001):

    with open('C:/config/data','rb') as f:
        max_data = pickle.load(f)
        min_data = pickle.load(f)

    model = build.build_model([1, hidden_dim, 1], freq_dim, learning_rate)
    # loading model
    model_path = model_path
    model.load_weights(model_path)

    # predition
    print('> Predicting... ')
    X_value = np.reshape(X_value,(X_value.shape[0],1))
    X_value = (2 * X_value - (max_data + min_data)) / (max_data - min_data)
    X_value = np.reshape(X_value,(X_value.shape[0], 1, 1))
    predicted = model.predict(X_value)
    # denormalization
    prediction = (predicted[:, :, 0] * (max_data - min_data) + (max_data + min_data)) / 2
    prediction = np.reshape(prediction,prediction.shape[0])

    return prediction

def getdiff(prediction, realdata, step):

    diff_data = prediction - realdata
    anomally_each = np.load("../dataset/science_anomally_each.npy")
    anomally_each = anomally_each[:,step:]
    anomally_all = np.load("../dataset/science_anomaly_all.npy")
    anomally_all = anomally_all[step:]

    high_limit = np.zeros((len(diff_data)), dtype=np.int32)
    low_limit = np.zeros((len(diff_data)), dtype=np.int32)
    for i in range(len(diff_data)):
        diff_dict = {}
        for j in range(len(diff_data[i])):
            diff_dict[diff_data[i][j]] = anomally_each[i][j]
        keys = sorted(diff_dict.keys())
        max_radio = 0
        high_limit_i = 0
        min_radio = 0
        low_limit_i = 0
        for j,key in enumerate(keys):
            less = [ 1 for k,v in diff_dict.items() if k<=key and v==1 ]
            less_radio = sum(less) / (j+1)
            more = [ 1 for k,v in diff_dict.items() if k>=key and v==1 ]
            more_radio = sum(more) / (len(keys)-j)
            if(less_radio >= min_radio):
                min_radio = less_radio
                low_limit_i = key
            if( more_radio > max_radio):
                max_radio = more_radio
                high_limit_i = key
        low_limit[i] = int(low_limit_i)
        high_limit[i] = int(high_limit_i)

    with open('./config/limit', 'wb') as f:
        pickle.dump(high_limit, f)
        pickle.dump(low_limit, f)


# Main Run Thread
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # n-step prediction
    parser.add_argument('-s', '--step', type=int, default=1)
    # data path
    parser.add_argument('-d', '--data_file', type=str, default='../dataset/science_data.npy')
    # dimension
    parser.add_argument('-hd', '--hidden_dim', type=int, default=50)
    parser.add_argument('-f', '--freq_dim', type=int, default=4)
    # training parameter
    parser.add_argument('-n', '--niter', type=int, default=4000)
    parser.add_argument('-ns', '--nsnapshot', type=int, default=20)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)

    args = parser.parse_args()
    step = args.step

    global_start_time = time.time()

    print('> Loading data... ')

    data_file = args.data_file
    X_train, y_train, X_val, y_val, X_test, y_test, gt_test, max_data, min_data = build.load_data(data_file, step)
    train_len = X_train.shape[1]
    val_len = X_val.shape[1] - X_train.shape[1]
    test_len = X_test.shape[1] - X_val.shape[1]

    """
    print('> Data Loaded. Compiling...')
    model = build.build_model([1, args.hidden_dim, 1], args.freq_dim, args.learning_rate)
    best_error = np.inf
    best_epoch = 0
    best_iter = 0


    
    start = time.time()
    for ii in range(int(args.niter / args.nsnapshot)):
        model.fit(
            X_train,
            y_train,
            batch_size=50,
            nb_epoch=args.nsnapshot,
            validation_split=0)

        num_iter = str(args.nsnapshot * (ii + 1))
        model.save_weights('./science_data/weights{}.hdf5'.format(num_iter), overwrite=True)

        predicted = model.predict(X_train)
        train_error = np.sum((predicted[:, :, 0] - y_train[:, :, 0]) ** 2) / (predicted.shape[0] * predicted.shape[1])

        print(num_iter, ' training error ', train_error)

        predicted = model.predict(X_val)
        print("predicted shape: ",predicted.shape)
        val_error = np.sum((predicted[:, -val_len:, 0] - y_val[:, -val_len:, 0]) ** 2) / (val_len * predicted.shape[0])

        print(' val error ', val_error)

        if (val_error < best_error):
            best_error = val_error
            best_iter = args.nsnapshot * (ii + 1)

    end = time.time()
    print('Training duration (s) : ', time.time() - global_start_time)
    print('best iteration ', best_iter)
    print('smallest error ', best_error)
    print('train time(s) : ', end - start)


    
    model = build.build_model([1, args.hidden_dim, 1], args.freq_dim, args.learning_rate)
    # loading model
    model_path = './science_data/weights{}.hdf5'.format(3660)
    model.load_weights(model_path)
    # predition
    print('> Predicting... ')
    start = time.time()
    predicted = model.predict(X_test)
    end = time.time()
    # denormalization
    prediction = (predicted[:, :, 0] * (max_data - min_data) + (max_data + min_data)) / 2

    error = np.sum((prediction[:, -test_len:] - gt_test[:, -test_len:]) ** 2) / (test_len * prediction.shape[0])
    print('The mean square error is: %f' % error)
    print('predict time: ', end - start)
    getdiff(prediction, gt_test , 1)

    
    tpr, fpr, auc = caulcute_tf(prediction,gt_test,test_len)
    all_data_tf = np.zeros((2,len(tpr)),dtype=np.float32)
    all_data_tf[0] = tpr
    all_data_tf[1] = fpr
    np.save('./Roc/SFM_ROC', all_data_tf)
    


    
    x = np.arange(0,prediction.shape[1]+step,1)
    with open("../dataset/nodename.pkl","rb") as f:
        nodenames = pickle.load(f)

    for ii in range(0, len(prediction)):
        plt.figure(facecolor='white')
        ax = plt.subplot(1,1,1)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.plot( x[0:-step], gt_test[ii, :], color='orangered', lw = 2 , label='Original Data')
        plt.plot( x[step:] , prediction[ii, :], color='orangered', linestyle = '--' ,lw = 2 ,label='Prediction Data')
        #if ii == len(prediction)-1:
        ax.set_xlabel('TIMESTEMPS', fontsize=10)
        ax.set_ylabel(nodenames[ii], fontsize=10)
        ax.legend(loc=2, fontsize=10)

        isExists = os.path.exists('./prediction')
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs('./prediction')
        plt.savefig('./prediction/{}.pdf'.format(nodenames[ii]))
        plt.show()
    """




