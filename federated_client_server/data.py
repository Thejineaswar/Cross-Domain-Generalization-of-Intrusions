import pandas as pd
import numpy as np
BASE_DIR = "../AE_Datasets/"
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

def print_summary():
    print("Started")
    for  i in DATA_LINK:
        df = pd.read_csv(BASE_DIR + i)
        if 'folds' not in df.columns:
            print(i)



def split_data(multi_round = True): #to split it into multiple rounds
    from sklearn.preprocessing import MinMaxScaler
    x_train_all = []
    y_train_all = []

    x_test_all = []
    y_test_all = []


    for i in range(len(DATA_LINK)):
        train = pd.read_csv(BASE_DIR + DATA_LINK[i] + "_train.csv")
        test = pd.read_csv(BASE_DIR + DATA_LINK[i] + "_valid.csv")
        scaler = MinMaxScaler()



        ytrain = train.iloc[:, -NUM_LABELS[i]:]
        ytest = test.iloc[:, -NUM_LABELS[i]:]
        xtrain = train.iloc[:, :-NUM_LABELS[i]]
        xtest = test.iloc[:, :-NUM_LABELS[i]]
        assert ytrain.shape[0] == xtrain.shape[0]
        assert ytest.shape[0] == xtest.shape[0]
        assert ytest.shape[1] == ytrain.shape[1]

        # print(f"X_train of {DATA_LINK[i]} is {xtrain.shape}")
        # print(f"X_test of {DATA_LINK[i]} is {xtest.shape}")
        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

        x_train_all.append(np.asarray(xtrain))
        x_test_all.append(np.asarray(xtest))

        y_train_all.append(np.asarray(ytrain))
        y_test_all.append(np.asarray(ytest))

    return x_train_all,x_test_all,y_train_all,y_test_all


if __name__ == "__main__":
    x_train_all,x_test_all,y_train_all,y_test_all = split_data()



