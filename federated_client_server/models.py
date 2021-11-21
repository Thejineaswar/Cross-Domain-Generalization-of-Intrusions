import tensorflow as tf
from tensorflow import keras

def MLP(weights=None):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[80, ]),
        keras.layers.Dense(200, activation='tanh'),
        keras.layers.Dense(100, activation='tanh'),
        keras.layers.Dense(56, activation='sigmoid')
    ])

    if weights is not None:
        model.set_weights(weights)

    return model


def get_model( mlp_weights=None):
    if mlp_weights is None:
        model = MLP()
    else:
        model = MLP(mlp_weights)
    return model




