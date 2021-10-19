import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# If you want the NN to figure out the label to neuron mapping, set this flag to true
BASE_DIR = "../Datasets/AE_formed_data/data"
AUTO_NEURON_MAPPING = True

if not AUTO_NEURON_MAPPING:
    CLIENT_SOFT_ALLOCATION = {
        0: {
            'name': "CICIDS 2018",
            'labels': 12,
            'start_index': 0
        },
        1: {
            'name': "CICIDS 2017",
            'labels': 15,
            'start_index': 12
        },
        2: {
            'name': "BOT IOT",
            'labels': 4,
            'start_index': 27
        },
        3: {
            'name': "NSL_KDD",
            'labels': 5,
            'start_index': 31
        },
        4: {
            'name': "TON_IOT",
            'labels': 10,
            'start_index': 36
        },
        5: {
            'name': "UNSW_NB15",
            'labels': 10,
            'start_index': 46
        }
    }
else:
    CLIENT_SOFT_ALLOCATION = {
        0: {
            'name': "CICIDS 2018",
            'labels': 12,
            'start_index': 0
        },
        1: {
            'name': "CICIDS 2017",
            'labels': 15,
            'start_index': 0
        },
        2: {
            'name': "BOT IOT",
            'labels': 4,
            'start_index': 0
        },
        3: {
            'name': "NSL_KDD",
            'labels': 5,
            'start_index': 0
        },
        4: {
            'name': "TON_IOT",
            'labels': 10,
            'start_index': 0
        },
        5: {
            'name': "UNSW_NB15",
            'labels': 10,
            'start_index': 0
        }
    }

def extract_labels(df, labels):
    temp = df.iloc[:, -labels:]
    labels = temp[temp == 1].stack().reset_index().drop(0, 1)
    labels = labels['level_1']
    return labels


TOT = 56

for i in CLIENT_SOFT_ALLOCATION.values():
    train = pd.read_csv(BASE_DIR + f"/{i['name']}_train.csv")
    test = pd.read_csv(BASE_DIR + f"/{i['name']}_valid.csv")

    train = extract_labels(train, i['labels'])
    test = extract_labels(test, i['labels'])

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit([[i] for i in train.values])
    train = enc.transform([[i] for i in train.values]).toarray()
    test = enc.transform([[i] for i in test.values]).toarray()

    train_labels = np.empty((train.shape[0], TOT))
    test_labels = np.empty((test.shape[0], TOT))

    lis = [0 for k in range(TOT)]

    for j in range(train.shape[0]):
        t = lis
        t[i['start_index']: i['start_index'] + i['labels']] = train[j]
        train_labels[j] = t
    for j in range(test.shape[0]):
        t = lis
        t[i['start_index']: i['start_index'] + i['labels']] = test[j]
        test_labels[j] = t
    print(test_labels.shape)
    print(train_labels.shape)
    with open("../Datasets/AE_formed_data/Label/" + i['name'] + '_train.npy', 'wb') as f:
        np.save(f, train_labels)

    with open("../Datasets/AE_formed_data/Label/" + i['name'] + '_test.npy', 'wb') as f:
        np.save(f, test_labels)