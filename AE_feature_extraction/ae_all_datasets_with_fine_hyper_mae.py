BASE_DIR = ""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import os
import tensorflow as tf

tf.random.set_seed(42)
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = "../Datasets/"
DATA_LINK = [
    'CICIDS_2018_folds.csv',
    'CICIDS_2017_folds.csv',
    'BOT_IOT_preprocessed.csv',
    'NSL_KDD_preprocessed.csv',
    'TON_IOT_preprocessd.csv',
    'UNSW_NB15_preprocessed.csv'
]

AE_HIDDEN_UNITS = 80

NUM_LABELS = [
    12,
    15,
    4,
    5,
    10,
    10
]



def Encoder(num_columns, num_labels,hidden_units, dropout_rates):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    # Encoder
    encoder = tf.keras.layers.GaussianNoise(dropout_rates[1])(x0)
    encoder = tf.keras.layers.Dropout(dropout_rates[2])(encoder)
    encoder = tf.keras.layers.Dense(hidden_units)(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('selu')(encoder)
    model = tf.keras.models.Model(inputs=inp, outputs=encoder)
    return model

    # Decoder


def Decoder(num_columns, num_labels, hidden_units, dropout_rates):
    inp = tf.keras.layers.Input(shape=(hidden_units))
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(inp)
    model = tf.keras.models.Model(inputs=inp, outputs=decoder)

    return model


def get_model(params_file, ae_weights=None, mlp_weights=None):
    x = tf.keras.layers.Input(shape=(params_file['num_columns'],))
    encoder = Encoder(**params_file)
    decoder = Decoder(**params_file)
    e_out = encoder(x)
    d_out = decoder(e_out)

    model = tf.keras.models.Model(inputs=x, outputs=d_out)

    ls = 0
    lr = 1e-3
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=[
                      tf.keras.losses.MeanAbsoluteError()
                  ]
                  )
    return model, encoder, decoder



model_params = {
    0: {
        'num_columns': 79,
        'num_labels': 12,
        'hidden_units': AE_HIDDEN_UNITS,
        'dropout_rates': [0.035, 0.0740089106066849, 0.044],
    },
    1: {
        'num_columns': 79,
        'num_labels': 15,
        'hidden_units': AE_HIDDEN_UNITS,
        'dropout_rates': [0.035, 0.001160893506920585, 0],
    },
    2: {
        'num_columns': 70,
        'num_labels': 4,
        'hidden_units': AE_HIDDEN_UNITS,
        'dropout_rates': [0.035, 0.23366970573577522, 0.378],
    },
    3: {
        'num_columns': 43,
        'num_labels': 5,
        'hidden_units': AE_HIDDEN_UNITS,
        'dropout_rates': [0.035, 0.019035611592714552, 0.513],
    },
    4: {
        'num_columns': 43,
        'num_labels': 10,
        'hidden_units': AE_HIDDEN_UNITS,
        'dropout_rates': [0.035, 0.12726406692524525, 0.222],
    },
    5: {
        'num_columns': 46,
        'num_labels': 10,
        'hidden_units': AE_HIDDEN_UNITS,
        'dropout_rates': [0.035, 0.027156361646183978, 0.367],
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

for i in range(len(DATA_LINK)):
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
    print(xtrain.shape)
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    model, encoder, decoder = get_model(model_params[i])
    batch_size = 64
    es = EarlyStopping(monitor='val_AE_ACC', min_delta=1e-4, patience=5, mode='max',
                       baseline=None, restore_best_weights=True, verbose=0)
    print(f"Starting training for {CLIENT_PRINT[i]}")
    history = model.fit(xtrain, [xtrain],
                        validation_data=(xtest, [xtest]),
                        epochs=20, batch_size=batch_size, callbacks=[es], verbose=True)
    print(f"Training for {CLIENT_PRINT[i]} Done")
    train_df = encoder.predict(xtrain)
    train_df = pd.DataFrame(data=train_df, index=[i for i in range(train_df.shape[0])],
                            columns=[i + 1 for i in range(train_df.shape[1])])
    train_df = pd.concat([train_df, ytrain.reset_index(drop=True)], axis=1)

    valid_df = encoder.predict(xtest)
    valid_df = pd.DataFrame(data=valid_df, index=[i for i in range(valid_df.shape[0])],
                            columns=[i + 1 for i in range(valid_df.shape[1])])
    valid_df = pd.concat([valid_df, ytest.reset_index(drop=True)], axis=1)

    # os.makedirs("../Datasets/AE_formed_data/labels")
    # os.makedirs("../Datasets/AE_formed_data/data")

    train_df.to_csv("../Datasets/AE_formed_data/data/" + f"{CLIENT_PRINT[i]}_train.csv", index=False)
    valid_df.to_csv("../Datasets/AE_formed_data/data/" + f"{CLIENT_PRINT[i]}_valid.csv", index=False)
    print(f"Features for {CLIENT_PRINT[i]} saved")
    # hist = pd.DataFrame(history.history)
    # hist.to_csv(f"../Datasets/AE_formed_data/{CLIENT_PRINT[i]}_AE.csv", index=False)