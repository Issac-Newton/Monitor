import numpy as np
import pandas as pd
import os
import pickle

directory = 'science_data'
filenames = os.listdir(directory)

all_data = np.zeros((len(filenames)+1, 2026), dtype=np.float32)
anomaly_data = np.zeros((1, 2026), dtype=np.float32)
anomaly_data_each = np.zeros((len(filenames),2026), dtype=np.float32)
file = []
for i in range(len(filenames)):
    filename = filenames[i]
    file.extend([filename])
    print(i)
    print(filename)

    data = pd.read_csv(directory + '/' + filename)
    vars = ['cpuutil']
    anomaly = ['label']
    data_vars = np.array(data[vars])
    data_anomaly = np.array(data[anomaly])
    data_vars = np.transpose(data_vars)

    anomaly_data = anomaly_data + np.transpose(data_anomaly)
    anomaly_data_each[i] = np.transpose(data_anomaly)
    all_data[i] = data_vars


with open("nodename.pkl","wb") as f:
    pickle.dump(file, f)


anomaly_data = np.array([1 if j == 2 else j for j in np.nditer(anomaly_data)])
all_data[-1] = anomaly_data
file.extend(['label_all'])
df = pd.DataFrame(all_data.transpose(), columns = file)
df.to_csv('science_data_all.csv')


all_data = np.delete(all_data,-1,0)
np.save('science_data', all_data)
np.save('science_anomaly_all',anomaly_data)
np.save('science_anomally_each',anomaly_data_each)
print(all_data.shape)
print(anomaly_data.shape)
print(anomaly_data_each.shape)