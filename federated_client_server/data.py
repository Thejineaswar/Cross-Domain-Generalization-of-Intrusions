import pandas as pd
import numpy as np

BASE_DIR = "../Datasets/AE_formed_data/Data/"
LABEL_DIR = "../Datasets/AE_formed_data/Label/"


DATA_LINK = [
    'CICIDS 2018',
    'CICIDS 2017',
    'BOT IOT',
    'NSL_KDD',
    'TON_IOT',
    'UNSW_NB15'
]

NUM_LABELS = [
    12,
    15,
    4,
    5,
    10,
    10
]


def split_data(DEBUG=False):  # to split it into multiple rounds
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    x_train_all = []
    y_train_all = []

    x_test_all = []
    y_test_all = []

    #     pca = []

    for i in range(len(DATA_LINK)):
        train = pd.read_csv(BASE_DIR + DATA_LINK[i] + "_train.csv")
        test = pd.read_csv(BASE_DIR + DATA_LINK[i] + "_valid.csv")
        if DEBUG:
            train = train.sample(320)
            test = test.sample(64)

        ytrain = np.load(LABEL_DIR +  f"{DATA_LINK[i]}_train.npy")
        ytest = np.load(LABEL_DIR + f"{DATA_LINK[i]}_test.npy")
        xtrain = train.iloc[:, :-NUM_LABELS[i]]
        xtest = test.iloc[:, :-NUM_LABELS[i]]
        assert ytrain.shape[0] == xtrain.shape[0]
        assert ytest.shape[0] == xtest.shape[0]

        x_train_all.append(np.asarray(xtrain))
        x_test_all.append(np.asarray(xtest))

        y_train_all.append(np.asarray(ytrain))
        y_test_all.append(np.asarray(ytest))

    return x_train_all, x_test_all, y_train_all, y_test_all


if __name__ == "__main__":
    x_train_all,x_test_all,y_train_all,y_test_all = split_data()
    print(x_train_all)



