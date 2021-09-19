#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: krishna
Base Source Code taken from : Krishna Repo Link
"""

import tensorflow as tf
from models import *
class Client:

    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate, weights, batch,params):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        # self.mini_batch=mini_batch
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate
        # self.decay_rate=decay_rate
        self.weights = weights
        self.batch = batch
        self.params = params

    def train(self):
        """
        # from federated_client_server.server import *
        """
        model,AE,MLP = get_model(params_file = self.params,
                                 ae_weights = None,
                                 mlp_weights = self.weights
                                 )

        # model.set_weights(self.weights)

        # getting the initial weight of the model
        # initial_weight=model.get_weights()
        # output_weight_list=[]

        # training the model
        # import animation
        # print('###### Client1 Training started ######')
        # wait=animation.Wait()
        # wait.start()

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate),
                      loss=[
                          tf.keras.losses.BinaryCrossentropy(),
                          tf.keras.losses.BinaryCrossentropy()
                      ],
                      metrics=[
                          tf.keras.metrics.Accuracy(name='AE_Accuracy'),
                          tf.keras.metrics.AUC(name='ANN_Accuracy'),
                      ]
                      )

        history = model.fit(
            self.dataset_x, [self.dataset_y,self.dataset_y],
            epochs=self.epoch_number,
            batch_size=self.batch
        )
        return MLP,AE








