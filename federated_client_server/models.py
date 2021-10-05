import tensorflow as tf

def MLP_with_weights(num_columns, num_labels, hidden_units, dropout_rates,weights):
    inp = tf.keras.layers.Input(shape=(128,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
    temp = tf.keras.models.Model(inputs = inp, outputs = x)
    print(len(weights))
    if(len(weights) == 30):
        temp.set_weights(weights[:-2])
    else:
        temp.set_weights(weights)

    out = tf.keras.layers.Dense(num_labels, activation='softmax', name='MLP')(temp.layers[-1].output)
    return tf.keras.models.Model(inputs=temp.input, outputs=[out])



def MLP(num_columns, num_labels, hidden_units, dropout_rates):
    inp = tf.keras.layers.Input(shape=(128,))
    x = tf.keras.layers.BatchNormalization()(inp)
    x = tf.keras.layers.Dropout(dropout_rates[3])(x)

    for i in range(2, len(hidden_units)):
        x = tf.keras.layers.Dense(hidden_units[i])(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('sigmoid')(x)
        x = tf.keras.layers.Dropout(dropout_rates[i + 2])(x)
    out = tf.keras.layers.Dense(num_labels, activation='softmax', name='MLP')(x)
    return tf.keras.models.Model(inputs=inp, outputs=[out])

def get_model(params_file, mlp_weights = None):
    
    x = tf.keras.layers.Input(shape=(128,))

    if mlp_weights is None:
        MLP_ = MLP(**params_file)
    else:
        params_copy = params_file.copy()
        params_copy['weights'] = mlp_weights
        MLP_ = MLP_with_weights(**params_copy)

    out_mlp = MLP_(x)
    model = tf.keras.models.Model(
        x, [
            out_mlp
        ]
    )
    return model




