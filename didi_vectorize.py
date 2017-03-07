import numpy as np
import pandas as pd
import gzip, h5py

############################
# BUILD SEQUENTIAL DATASET #
############################
def build_sequential_data(df, targ, n_samples, seq_len, n_features):
    seq_data = np.zeros((n_samples, seq_len, n_features))
    seq_targ = np.zeros((n_samples, seq_len, 1))

    seq_data[2:] = np.dstack((df[:-2], df[1:-1], df[2:])).swapaxes(-1,-2)
    seq_targ[2:] = np.dstack((targ[:-2], targ[1:-1], targ[2:])).swapaxes(-1,-2)

    for i in range(2):
        seq_data[i, -(i+1):] = seq_data[2, :(i+1)]
        seq_targ[i, -(i+1):] = seq_targ[2, :(i+1)]

    train_indicies = pd.read_csv('training-07.csv', header=None).values.flatten()-1
    test_indicies = pd.read_csv('test-07.csv', header=None).values.flatten()-1
    np.random.shuffle(train_indicies)
    np.random.shuffle(test_indicies)

    train_set = [seq_data[train_indicies], seq_targ[train_indicies]]
    valid_set = [seq_data[test_indicies], seq_targ[test_indicies]]
    test_set = [seq_data[test_indicies], seq_targ[test_indicies]]

    print("... compressing")
    h5f = h5py.File('didi_train_rnn.h5', 'w')
    h5f.create_dataset('train_set_x', data=train_set[0], compression="gzip")
    h5f.create_dataset('train_set_y', data=train_set[1], compression="gzip")
    h5f.create_dataset('valid_set_x', data=valid_set[0], compression="gzip")
    h5f.create_dataset('valid_set_y', data=valid_set[1], compression="gzip")
    h5f.create_dataset('test_set_x', data=test_set[0], compression="gzip")
    h5f.create_dataset('test_set_y', data=test_set[1], compression="gzip")
    h5f.close()


##########################
# BUILD STANDARD DATASET #
##########################
def build_standard_data(df, targ, n_samples, n_features):
    train_indicies = pd.read_csv('training-07.csv', header=None).values.flatten()-1
    test_indicies = pd.read_csv('test-07.csv', header=None).values.flatten()-1
    np.random.shuffle(train_indicies)
    np.random.shuffle(test_indicies)

    train_set = [df[train_indicies], targ[train_indicies]]
    valid_set = [df[test_indicies], targ[test_indicies]]
    test_set = [df[test_indicies], targ[test_indicies]]

    print("... compressing")
    h5f = h5py.File('didi_train.h5', 'w')
    h5f.create_dataset('train_set_x', data=train_set[0], compression="gzip")
    h5f.create_dataset('train_set_y', data=train_set[1], compression="gzip")
    h5f.create_dataset('valid_set_x', data=valid_set[0], compression="gzip")
    h5f.create_dataset('valid_set_y', data=valid_set[1], compression="gzip")
    h5f.create_dataset('test_set_x', data=test_set[0], compression="gzip")
    h5f.create_dataset('test_set_y', data=test_set[1], compression="gzip")
    h5f.close()


data_file = 'didi_train.csv'
data_csv = pd.read_csv(data_file)

data = {}
n_features = 0

columns =  ['district_id', 'dow', 'hour', 'minute', 'price_avg', 'price_median', 'price_max', 'destinations', 'tj1', 'tj2', 'tj3', 'tj4', 'tjg1', 'tjg2', 'tjg3', 'tjg4', 'weather', 'temperature', 'pollutants']

for column in columns:
    data[column] = data_csv[column].values.T
    size = np.ceil(np.log2(np.max(data[column])-np.min(data[column]))+1)
    n_features += size

n_samples = data_csv.shape[0]
seq_len = 3
n_features = n_features.astype('int32')
df = np.zeros((n_samples, n_features))
index = 0

for column in columns:
    data[column] = np.array([data_csv[column].values]).T
    size = np.ceil(np.log2(np.max(data[column])-np.min(data[column]))+1)
    size = size.astype('int32')
    df[:,index: index+size] = ((data[column] & (1 << np.arange(size))) > 0)
    index += size
    
print('data shape', df.shape)

targ = np.array([data_csv['demand'].values]).T
print("... data built ... dividing dataset")

build_sequential_data(df, targ, n_samples, seq_len, n_features)
build_standard_data(df, targ, n_samples, n_features)


# df[:,0:4] = ((data['dow'] & (1 << np.arange(4)))>0)
# df[:,4:9] = ((hour & (1 << np.arange(5)))>0)
# df[:,9:12] = ((minute & (1 << np.arange(3)))>0)
# df[:,12:21] = ((price_avg & (1 << np.arange(9)))>0)
# df[:,21:30] = ((price_median & (1 << np.arange(9)))>0)
# df[:,30:41] = ((price_max & (1 << np.arange(11)))>0)
# df[:,41:48] = ((destinations & (1 << np.arange(7)))>0)
# df[:,48:55] = ((tj1 & (1 << np.arange(7)))>0)
# df[:,55:62] = ((tj2 & (1 << np.arange(7)))>0)
# df[:,62:69] = ((tj3 & (1 << np.arange(7)))>0)
# df[:,69:76] = ((tj4 & (1 << np.arange(7)))>0)
# df[:,76:83] = ((tjg1 & (1 << np.arange(7)))>0)
# df[:,83:90] = ((tjg2 & (1 << np.arange(7)))>0)
# df[:,90:97] = ((tjg3 & (1 << np.arange(7)))>0)
# df[:,97:104] = ((tjg4 & (1 << np.arange(7)))>0)
# df[:,104:108] = ((weather & (1 << np.arange(4)))>0)
# df[:,108:114] = ((temperature & (1 << np.arange(6)))>0)
# df[:,114:123] = ((pollutants & (1 << np.arange(9)))>0)
# df[:,123:130] = ((district_id & (1 << np.arange(7)))>0)
