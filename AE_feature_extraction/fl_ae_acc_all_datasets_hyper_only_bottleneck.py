
import pandas as pd
import pickle
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

tf.random.set_seed(42)
import tensorflow.keras.backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

# !pip install optuna
import optuna


def Encoder(num_columns, num_labels, hidden_units, dropout_rates):  # Hidden units to be suggested by optuna
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units)(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('selu')(encoder)

    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)

    x_ae = tf.keras.layers.Dense(128)(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('selu')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)
    out_ae = tf.keras.layers.Dense(num_labels, activation='softmax', name='ae_out')(x_ae)

    model = tf.keras.models.Model(inputs=inp, outputs=out_ae)

    return model


def get_model(params_file, ae_weights=None, mlp_weights=None):
    x = tf.keras.layers.Input(shape=(params_file['num_columns'],))
    noise = tf.keras.layers.GaussianNoise(params_file['dropout_rates'][0])(x)
    #     params_file["hidden_units"] = optuna_suggest
    encoder = Encoder(**params_file)
    #     decoder = Decoder(**params_file)
    e_out = encoder(noise)
    #     d_out = decoder(e_out)

    model = tf.keras.models.Model(inputs=x, outputs=e_out)

    ls = 0
    lr = 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=[
                      tf.keras.losses.BinaryCrossentropy(label_smoothing=ls)
                  ],
                  metrics=[
                      tf.keras.metrics.CategoricalAccuracy(name='AE_ACC')
                  ],
                  )
    return model, encoder


BASE_DIR = "../Datasets/"
DATA_LINK = [
    'CICIDS_2018_folds.csv',
    'CICIDS_2017_folds.csv',
    'BOT_IOT_preprocessed.csv',
    'NSL_KDD_preprocessed.csv',
    'TON_IOT_preprocessd.csv',
    'UNSW_NB15_preprocessed.csv'
]

NUM_LABELS = [
    12,
    15,
    4,
    5,
    10,
    10
]

model_params = {
    0: {
        'num_columns': 79,
        'num_labels': 12,
        'hidden_units': [128, 128, 1024, 512, 512, 256],
        'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
    },
    1: {
        'num_columns': 79,
        'num_labels': 15,
        'hidden_units': [128, 128, 1024, 512, 512, 256],
        'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
    },
    2: {
        'num_columns': 70,
        'num_labels': 4,
        'hidden_units': [128, 128, 1024, 512, 512, 256],
        'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
    },
    3: {
        'num_columns': 43,
        'num_labels': 5,
        'hidden_units': [128, 128, 1024, 512, 512, 256],
        'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
    },
    4: {
        'num_columns': 43,
        'num_labels': 10,
        'hidden_units': [128, 128, 1024, 512, 512, 256],
        'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
    },
    5: {
        'num_columns': 46,
        'num_labels': 10,
        'hidden_units': [128, 128, 1024, 512, 512, 256],
        'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
    },
}

CLIENT_PRINT = {
    0: "CICIDS 2018",
    1: "CICIDS 2017",
    2: "BOT IOT",
    3: "NSL_KDD",
    4: "TON_IOT",
    5: "UNSW_NB15"
}

import pickle

for i in range(len(DATA_LINK)):

    def objective(trial):
        tf.keras.backend.clear_session()

        bottleneck = trial.suggest_int('intermediate', 32, 128)

        fold = 5
        print(CLIENT_PRINT[i])
        df = pd.read_csv(BASE_DIR + DATA_LINK[i])

        test = df[df['folds'] == fold]
        train = df[df['folds'] != fold]

        ytrain = train.iloc[:, -NUM_LABELS[i]:]
        ytest = test.iloc[:, -NUM_LABELS[i]:]
        xtrain = train.iloc[:, :-NUM_LABELS[i]]
        xtest = test.iloc[:, :-NUM_LABELS[i]]
        assert ytrain.shape[0] == xtrain.shape[0]
        scaler = MinMaxScaler()

        xtrain = scaler.fit_transform(xtrain)
        xtest = scaler.transform(xtest)

        model_params[i]['hidden_units'] = bottleneck
        model, encoder = get_model(model_params[i])
        batch_size = 64
        print(f"Starting training for {CLIENT_PRINT[i]}")
        history = model.fit(xtrain, [ytrain],
                            validation_data=(xtest, [ytest]),
                            epochs=15, batch_size=batch_size, verbose=True)
        score = model.evaluate(xtest, ytest, verbose=1)
        return score[1]


    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best trial number:", study.best_trial.number)
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    with open(f'AE_BEST_PARAMS_ACC/{DATA_LINK[i]}_study.pickle', 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
    del study