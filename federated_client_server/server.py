import numpy as np
import tensorflow.keras as keras
from .client import Client

def get_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[122, ]),
        keras.layers.Dense(200, activation='tanh'),
        keras.layers.Dense(100, activation='tanh'),
        keras.layers.Dense(5, activation='softmax')
    ])

    return model


def model_average(client_weights):
    average_weight_list = []
    for index1 in range(len(client_weights[0])):
        layer_weights = []
        for index2 in range(len(client_weights)):
            weights = client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight = np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list


def create_model():
    model = get_model()
    weight = model.get_weights()
    return weight


def getdata():
    """
    Read all the datasets here and append it to a list. The variable y_data and x_data only contain train data.

    :return: x_data,y_data, X_valid and y_valid
    """
    raise NotImplementedError


def evaluate_model(accuracy_list, weight, learning_rate, X_valid,y_valid):
    model = get_model()
    model.set_weights(weight)
    model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=keras.optimizers.SGD(lr=learning_rate),
            metrics=['accuracy']
    )

    result = model.evaluate(X_valid, y_valid)

    if len(accuracy_list) == 0:
        accuracy_list.append(0)
        if result[1] > accuracy_list[len(accuracy_list) - 1]:
            return True, result[1]

    elif result[1] > accuracy_list[len(accuracy_list) - 1]:
        return True, result[1]
    else:
        return False, result[1]



def train_server(training_rounds, epoch, batch, learning_rate):
    # temp_variable
    # training_rounds=2 
    # epoch=5 
    # batch=128

    accuracy_list = []
    client_weight_for_sending = []

    x_data,y_data,X_valid, y_valid = getdata()

    for index1 in range(1, training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged = []
        for index in range(len(y_data)):
            print('-------Client-------', index)
            if index1 == 1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight = create_model()
                client = Client(x_data[index], y_data[index], epoch, learning_rate, initial_weight, batch)
                weight = client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client = Client(x_data[index], y_data[index], epoch, learning_rate,
                                client_weight_for_sending[index1 - 2], batch)
                weight = client.train()
                client_weights_tobe_averaged.append(weight)

        # calculating the avearge weight from all the clients
        client_average_weight = model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)

        # validating the model with avearge weight
        model = get_model()

        model.set_weights(client_average_weight)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=learning_rate),
                      metrics=['accuracy'])
        result = model.evaluate(X_valid, y_valid)
        accuracy = result[1]
        print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)

    return accuracy_list


def train_server_weight_discard(training_rounds, epoch, batch, learning_rate):
    # temp_variable
    # training_rounds=5
    # epoch=3
    # batch=64 
    # learning_rate=0.01

    x_data, y_data, _, _ = getdata()

    accuracy_list = []
    client_weight_for_sending = []

    for index1 in range(1, training_rounds):
        print('Training for round ', index1, 'started')
        client_weights_tobe_averaged = []
        for index in range(len(y_data)):
            print('-------Client-------', index)
            if index1 == 1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight = create_model()
                client = Client(x_data[index], y_data[index], epoch, learning_rate, initial_weight, batch)
                weight = client.train()
                client_weights_tobe_averaged.append(weight)
            else:
                client = Client(x_data[index], y_data[index], epoch, learning_rate,
                                client_weight_for_sending[index1 - 2], batch)
                weight = client.train()
                client_weights_tobe_averaged.append(weight)

        # calculating the avearge weight from all the clients
        client_average_weight = model_average(client_weights_tobe_averaged)
        boolean, accuracy = evaluate_model(accuracy_list, client_average_weight, learning_rate)
        if boolean == True:
            client_weight_for_sending.append(client_average_weight)
            print('#######-----Acccuracy for round ', index1, 'is ', accuracy, ' ------########')
            accuracy_list.append(accuracy)

        else:
            print('Weight discarded due to low accuarcy')
            client_weight_for_sending.append(client_weight_for_sending[len(client_weight_for_sending) - 1])
            accuracy_list.append(accuracy_list[len(accuracy_list) - 1])

    return accuracy_list, client_weight_for_sending
