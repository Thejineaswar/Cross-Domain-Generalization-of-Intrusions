import tensorflow as tf

def Autoencoder(num_columns, num_labels, hidden_units, dropout_rates):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x0 = tf.keras.layers.BatchNormalization()(inp)

    encoder = tf.keras.layers.GaussianNoise(dropout_rates[0])(x0)
    encoder = tf.keras.layers.Dense(hidden_units[0])(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.layers.Activation('swish')(encoder)

    # Decoder
    decoder = tf.keras.layers.Dropout(dropout_rates[1])(encoder)
    decoder = tf.keras.layers.Dense(num_columns, name='decoder')(decoder)

    x_ae = tf.keras.layers.Dense(hidden_units[1])(decoder)
    x_ae = tf.keras.layers.BatchNormalization()(x_ae)
    x_ae = tf.keras.layers.Activation('swish')(x_ae)
    x_ae = tf.keras.layers.Dropout(dropout_rates[2])(x_ae)
    out_ae = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='AE')(x_ae)
    return tf.keras.models.Model(inputs=inp, outputs=[out_ae, encoder])


def MLP(num_columns, num_labels, hidden_units, dropout_rates):
    inp = tf.keras.layers.Input(shape=(148,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('swish')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
    out = tf.keras.layers.Dense(num_labels, activation='sigmoid', name='MLP')(x)
    return tf.keras.models.Model(inputs=inp, outputs=[out])

def get_model():
    params_17 = {
                    'num_columns': 20,
                    'num_labels': 15,
                    'hidden_units': [128, 128, 1024, 512, 512, 256],
                    'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
                 }
    
    x = tf.keras.layers.Input(shape=(params_17['num_columns'],))
    noise = tf.keras.layers.GaussianNoise(params_17['dropout_rates'][0])(x)
    AE = Autoencoder(**params_17)
    out_ae, encoder = AE(noise)
    MLP_ = MLP(**params_17)
    out_mlp = MLP_(tf.keras.layers.Concatenate()([x, encoder]))
    model = tf.keras.models.Model(
        x, [
            out_ae,
            out_mlp
        ]
    )

