import pandas as pd
import numpy as np
BASE_DIR = "../Datasets/"
DATA_LINK = [
    'CICIDS_2018_folds.csv',
    'CICIDS_2017_folds.csv'
]

NUM_LABELS = [
    12,
    15
]

def split_data(multi_round = True): #to split it into multiple rounds
    from sklearn.decomposition import PCA

    x_train_all = []
    y_train_all = []

    x_test_all = []
    y_test_all = []

    pca = []

    for i in range(len(DATA_LINK)):
        df = pd.read_csv(BASE_DIR + DATA_LINK[i])
        df = df.reset_index(drop = True)
        test = df[df['folds'] == 5]
        train = df[df['folds'] != 5]#df[df['folds'] != 5]

        # if folds ==0:
        #     train = train.loc[train['folds'].isin([i for i in range(0,5)])]
        #     pca_ = PCA(n_components=20, random_state=42)
        # else :
        #     train = train.loc[train['folds'].isin([i for i in range(6,10)])]


        pca_ = PCA(n_components=20, random_state=42)

        ytrain = train.iloc[:, -NUM_LABELS[i]:]
        ytest = test.iloc[:, -NUM_LABELS[i]:]
        xtrain = train.iloc[:, :-NUM_LABELS[i]]
        xtest = test.iloc[:, :-NUM_LABELS[i]]
        assert ytrain.shape[0] == xtrain.shape[0]
        print(f"X_train {xtrain.shape}")
        xtrain = pca_.fit_transform(xtrain)
        xtest = pca_.transform(xtest)

        x_train_all.append(np.asarray(xtrain))
        x_test_all.append(np.asarray(xtest))

        y_train_all.append(np.asarray(ytrain))
        y_test_all.append(np.asarray(ytest))

        pca.append(pca_)

    return x_train_all,x_test_all,y_train_all,y_test_all


if __name__ == "__main__":
    x_train_all,x_test_all,y_train_all,y_test_all = split_data()
    print(x_train_all[1].shape)



