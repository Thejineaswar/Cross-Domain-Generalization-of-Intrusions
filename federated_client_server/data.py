import pandas as pd

BASE_DIR = "../Datasets/"
DATA_LINK = [
    'CICIDS_2018_folds.csv',
    'CICIDS_2017_folds.csv'
]

NUM_LABELS = [
    15,
    12   
]


def split_data():
    from sklearn.decomposition import PCA

    x_train_all = []
    y_train_all = []

    x_test_all = []
    y_test_all = []

    pca = []

    for i in range(len(DATA_LINK)):
        df = pd.read_csv(BASE_DIR + i)
        test = df[df['folds'] != 5]
        train = df[df['folds'] == 5]
        pca_ = PCA(n_components=20, random_state=42)

        ytrain = train.iloc[:, -NUM_LABELS[i]:]
        ytest = test.iloc[:, -NUM_LABELS[i]:]
        xtrain = train.iloc[:, :-NUM_LABELS[i]]
        xtest = test.iloc[:, :-NUM_LABELS[i]]


        xtrain = pca_.fit_transform(xtrain)
        xtest = pca_.transform(xtest)

        x_train_all.append(xtrain)
        x_test_all.append(xtest)

        y_train_all.append(ytrain)
        y_test_all.append(ytest)

        pca.append(pca_)

    return x_train_all,x_test_all,y_train_all,y_test_all



