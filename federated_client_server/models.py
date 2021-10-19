import tensorflow as tf


def MLP(num_columns, num_labels, hidden_units, dropout_rates, weights=None):
    inp = tf.keras.layers.Input(shape=(num_columns,))
    x = tf.keras.layers.Dense(hidden_units[2])(inp)
    x = tf.keras.layers.Activation('selu')(x)

    x = tf.keras.layers.Dense(hidden_units[3])(x)
    x = tf.keras.layers.Activation('selu')(x)

    x = tf.keras.layers.Dense(hidden_units[4])(x)
    x = tf.keras.layers.Activation('selu')(x)

    x = tf.keras.layers.Dense(hidden_units[5])(x)
    x = tf.keras.layers.Activation('selu')(x)

    out = tf.keras.layers.Dense(num_labels, name='MLP')(x)
    out = tf.keras.layers.Activation('sigmoid')(out)

    model = tf.keras.models.Model(inputs=inp, outputs=[out])
    if weights is not None:
        model.set_weights(weights)

    return model


def get_model(params_file, mlp_weights=None):
    x = tf.keras.layers.Input(shape=(params_file["num_columns"],))
    params_copy = params_file.copy()
    if mlp_weights is None:
        params_copy['weights'] = None
        MLP_ = MLP(**params_copy)
    else:
        params_copy = params_file.copy()
        params_copy['weights'] = mlp_weights
        #         if last_dense is not None:
        #             params_copy['last_dense'] = last_dense
        #         else:
        #             params_copy['last_dense'] = None
        MLP_ = MLP(**params_copy)

    out_mlp = MLP_(x)
    model = tf.keras.models.Model(
        x, [
            out_mlp
        ]
    )
    return model




