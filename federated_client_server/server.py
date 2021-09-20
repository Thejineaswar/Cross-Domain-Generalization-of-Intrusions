import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
from client import Client

from models import *
from data import *
from model_params import *

def model_average(client_weights):
    average_weight_list = []
    for index1 in range(len(client_weights[0])):
        layer_weights = []
        for index2 in range(len(client_weights)-2):
            weights = client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight = np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list


def create_model(params,ae_weights = None,mlp_weights = None):
    model, AE,MLP_ = get_model(params)
    weight = MLP_.get_weights()
    return weight



CLIENT_PRINT = {
    0: "CICIDS 2018",
    1: "CICIDS 2017"
}

PARAMS = get_model_params()

def train_server(training_rounds, epoch, batch, learning_rate):
    # training_rounds=2
    # epoch=5
    # batch=128

    accuracy_list = []
    client_weight_for_sending = []

    x_data,x_test,y_data,y_test = split_data()

    for index1 in range(1, training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged = []
        client_ae_weights = []
        for index in range(len(y_data)):
            print('-------Client-------', CLIENT_PRINT[index])
            if index1 == 1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight = create_model(PARAMS[index])
                client = Client(
                        x_data[index],
                        y_data[index],
                        epoch,
                        learning_rate,
                        initial_weight,
                        batch,
                    PARAMS[index]
                    )
                AE_weights, MLP_weights = client.train()
                client_weights_tobe_averaged.append(MLP_weights.get_weights())
                client_ae_weights.append(AE_weights.get_weights())
            else:
                client = Client(x_data[index],
                                y_data[index],
                                epoch,
                                learning_rate,
                                client_weight_for_sending[index1 - 2],
                                batch,
                                PARAMS[index]) # why minus 2?
                AE_weights, MLP_weights = client.train()
                client_weights_tobe_averaged.append(MLP_weights.get_weights())
                client_ae_weights.append(AE_weights.get_weights())

        client_average_weight = model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)

        # validating the model with avearge weight
        print(f"Evaluation for round{index1}:")
        print(f"Number of client weights : {len(client_ae_weights)}")
        for index in range(len(y_test)):
            model = get_model(PARAMS[index] ,ae_weights = client_ae_weights[index], mlp_weights = client_average_weight)

            model.compile(
                loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.SGD(lr=learning_rate),
                          metrics=['accuracy'])
            result = model.evaluate(x_test[index], y_test[index])
            print(result)
            accuracy = result[1]
            print(f"###### Accuracy for {CLIENT_PRINT[index]} -> {result}")
            #print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
            accuracy_list.append(accuracy)

    return accuracy_list


if __name__ == '__main__':
    training_accuracy_list100 = train_server(
                                                training_rounds=100,
                                                epoch=1,
                                                batch=32,
                                                learning_rate=0.01
                                             )


