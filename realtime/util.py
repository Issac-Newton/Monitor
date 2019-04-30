import numpy as np
import sys
import os.path
##sys.path.append('C:/Users/Shaohan/PycharmProjects/ScienceProject/ScienceProject/catalog/SFM')
from .SFM.train.train import predict
import pickle



def from_json_to_dict(jsdata):
    jsdata = jsdata
    status = 0
    dict = {'casnw':0, 'dicp':0, 'era':0, 'erai':0, 'gspcc':0 , 'hku':0, 'hust':0, 'iapcm':0, 'nscccs':0, 'nsccgz':0, 'nsccjn':0, 'nscctj':0, 'nsccwx':0, 'siat':0, 'sjtu':0, 'ssc':0, 'ustc':0, 'xjtu':0}
    if jsdata['status_code'] != 0:
        ##print('error')
        status = 3
        return dict,status

    mapdata = jsdata['mapdata']
    profiles = mapdata['profiles']
    #print(profiles)

    status = 0
    return profiles,status
    '''
    for profile in profiles:
        cpuutil = 0.0
        nodename = ''
        for k, v in profile.items():
            if k == 'nodeName':
                nodename = v
            elif k == 'cpuutil' :
                if v != None:
                    cpuutil = v
                else:
                    cpuutil = 0
                    status = 2
        if nodename in dict.keys():
            dict[nodename] = cpuutil
    return dict,status
    '''
def getstatus(dict_time , predict_value):
    dict = dict_time
    time_value = np.zeros(len(dict),dtype = np.float32)
    # 保证模型的顺序
    time_value[0] = dict['casnw']
    time_value[1] = dict['dicp']
    time_value[2] = dict['era']
    time_value[3] = dict['erai']
    time_value[4] = dict['gspcc']
    time_value[5] = dict['hku']
    time_value[6] = dict['hust']
    time_value[7] = dict['iapcm']
    time_value[8] = dict['nscccs']
    time_value[9] = dict['nsccgz']
    time_value[10] = dict['nsccjn']
    time_value[11] = dict['nscctj']
    time_value[12] = dict['nsccwx']
    time_value[13] = dict['siat']
    time_value[14] = dict['sjtu']
    time_value[15] = dict['ssc']
    time_value[16] = dict['ustc']
    time_value[17] = dict['xjtu']

    predict_time_value = predict(time_value)
    diff_data = predict_value - time_value

    file = os.path.join(os.path.dirname(__file__),'SFM/train/config/limit')

    with open(file,'rb') as f:
        high_limit = pickle.load(f)
        low_limit = pickle.load(f)
    status = 0
    for i in range(len(diff_data)):
        if diff_data[i] < low_limit[i] and diff_data[i] > high_limit[i] :
            status = 1

    return time_value,predict_time_value,status