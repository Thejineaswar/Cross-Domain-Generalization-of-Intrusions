import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import pickle

from client import Client

from models import *
from data import *
from model_params import *


def model_average(client_weights):
    average_weight_list = []
    for index1 in range(len(client_weights[0]) - 2): # -2 to exclude softmax dense
        layer_weights = []
        for index2 in range(len(client_weights)):
            weights = client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight = np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list


def create_model(params,ae_weights = None,mlp_weights = None):
    model, AE,MLP_ = get_model(params)
    ann_weight = MLP_.get_weights()
    ae_weight = AE.get_weights()
    return ann_weight,ae_weight



CLIENT_PRINT = {
    0: "CICIDS 2018",
    1: "CICIDS 2017",
    2:  "BOT IOT",
    3: "NSL_KDD",
    4:"TON_IOT",
    5:"UNSW_NB15"
}

PARAMS = get_model_params()

def train_server(training_rounds, epoch, batch, learning_rate):
    # training_rounds=2
    # epoch=5
    # batch=128

    accuracy_list = []
    client_weight_for_sending = []
    client_ae_for_sending = [i for i in range(len(CLIENT_PRINT.keys()))]

    x_data,x_test,y_data,y_test = split_data()

    for index1 in range(1, training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged = []
        for index in range(len(y_data)):
            print('-------Client-------', CLIENT_PRINT[index])
            if index1 == 1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight,ae_weight= create_model(PARAMS[index])
                client = Client(
                        x_data[index],
                        y_data[index],
                        epoch,
                        learning_rate,
                        initial_weight,
                        ae_weight,
                        batch,
                    PARAMS[index]
                    )
                AE_weights, MLP_weights = client.train()
                # print(MLP_weights.get_weights())
                client_weights_tobe_averaged.append(MLP_weights.get_weights())
                # client_ae_weights.append(AE_weights.get_weights())
                client_ae_for_sending[index] = AE_weights.get_weights()
            else:
                client = Client(x_data[index],
                                y_data[index],
                                epoch,
                                learning_rate,
                                client_weight_for_sending[index1 - 2],
                                client_ae_for_sending[index],
                                batch,
                                PARAMS[index]) # why minus 2?
                AE_weights, MLP_weights = client.train()
                client_weights_tobe_averaged.append(MLP_weights.get_weights())
                # client_ae_weights.append(AE_weights.get_weights())
                client_ae_for_sending[index] = AE_weights.get_weights()

        client_average_weight = model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)

        # validating the model with avearge weight
        print(f"Evaluation for round{index1}:")
        for index in range(len(y_test)):
            model, _,_ = get_model(PARAMS[index] ,ae_weights = client_ae_for_sending[index], mlp_weights = client_average_weight)

            model.compile(
                          loss=[
                              tf.keras.losses.BinaryCrossentropy(),
                              tf.keras.losses.BinaryCrossentropy()
                          ],
                          metrics=[
                              tf.keras.metrics.CategoricalAccuracy(name='AE_Accuracy'),
                              tf.keras.metrics.CategoricalAccuracy(name='ANN_Accuracy'),
                          ]
                          )
            print("Evaluation Started")
            result = model.evaluate(x_test[index], [y_test[index],y_test[index]],verbose = False)
            print(result)
            accuracy = result[1]
            print(f"###### Accuracy for {CLIENT_PRINT[index]} -> {result}")
            #print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
            accuracy_list.append(accuracy)

    return accuracy_list


if __name__ == '__main__':
    training_accuracy_list = train_server(
                                                training_rounds=150,
                                                epoch=1,
                                                batch=32,
                                                learning_rate=0.001
                                             )
    with open('accuracy_list.txt','wb') as fp:
        pickle.dump(training_accuracy_list,fp)


