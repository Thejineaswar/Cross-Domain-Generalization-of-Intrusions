def get_model_params():
    params = {
        0 : {
            'num_columns': 79,
            'num_labels': 12,
            'hidden_units': [128, 128, 1024, 512, 512, 256],
            'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
        1: {
            'num_columns': 79,
             'num_labels': 15,
             'hidden_units': [128, 128, 1024, 512, 512, 256],
             'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
        2: {
            'num_columns': 70,
            'num_labels': 4,
            'hidden_units': [128, 128, 1024, 512, 512, 256],
            'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
        3: {
            'num_columns': 43,
            'num_labels': 5,
            'hidden_units': [128, 128, 1024, 512, 512, 256],
            'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
        4: {
            'num_columns': 43,
            'num_labels': 10,
            'hidden_units': [128, 128, 1024, 512, 512, 256],
            'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
        5: {
            'num_columns': 46,
            'num_labels': 10,
            'hidden_units': [128, 128, 1024, 512, 512, 256],
            'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
    }

    return params