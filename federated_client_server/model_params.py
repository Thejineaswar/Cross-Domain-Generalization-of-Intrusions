def get_model_params():
    params = {
        0 : {
            'num_columns': 20,
            'num_labels': 12,
            'hidden_units': [128, 128, 1024, 512, 512, 256],
            'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        },
        1: {
            'num_columns': 20,
             'num_labels': 15,
             'hidden_units': [128, 128, 1024, 512, 512, 256],
             'dropout_rates': [0.035, 0.038, 0.42, 0.10, 0.49, 0.32, 0.27, 0.43],
        }
    }

    return params