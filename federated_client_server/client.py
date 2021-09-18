#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: krishna
Base Source Code taken from : Krishna Repo Link
"""
from .models import *

class Client:

    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate, weights, batch):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        # self.mini_batch=mini_batch
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate
        # self.decay_rate=decay_rate
        self.weights = weights
        self.batch = batch

    def train(self):
        import numpy as np
        import pandas as pd
        import matplotlib as plt
        from tensorflow import keras
        from .models import *
        from .server import *

        model,AE,MLP = get_model()

        # setting weight of the model
        model.set_weights(self.weights)

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
                          tf.keras.losses.SparseCategoricalCrossentropy(),
                          tf.keras.losses.SparseCategoricalCrossentropy()
                      ],
                      metrics=[
                          tf.keras.metrics.Accuracy(name='AE_Accuracy'),
                          tf.keras.metrics.AUC(name='ANN_Accuracy'),
                      ]
                      )

        history = model.fit(
            self.dataset_x, self.dataset_y,
            epochs=self.epoch_number,
            batch_size=self.batch
        )

        return MLP








