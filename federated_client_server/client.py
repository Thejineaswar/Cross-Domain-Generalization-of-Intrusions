#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: krishna
Base Source Code taken from :
https://github.com/cryptogeekk/Detecting-zero-days-attack-with-Federated-Learning/blob/master/client.py
"""

import tensorflow as tf
from models import *
from tqdm.keras import TqdmCallback
class Client:

    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate, mlp_weights, batch):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate
        self.weights = mlp_weights
        self.batch = batch

    def train(self):
        """
        # from federated_client_server.server import *
        """
        model = get_model(
                            mlp_weights = self.weights
                        )
        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                      loss=[
                          tf.keras.losses.BinaryCrossentropy()
                      ],
                      metrics=[
                          tf.keras.metrics.CategoricalAccuracy(name='ANN_Accuracy'),
                      ]
                      )

        history = model.fit(
            self.dataset_x, [self.dataset_y],
            epochs=self.epoch_number,
            batch_size=self.batch,
            verbose = 0,
            callbacks=[TqdmCallback(verbose=2)]
        )
        return model








