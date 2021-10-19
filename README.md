# IoT-IDS-using-Federated-Learning
### Instructions to run the code
* To get the Autoencoder processed dataset, run the `AE_feature_extraction/ae_all_datasets_with_hyper_mae.py"
  * If you need to change the dimensions of AE bottleneck layer change the value of the variable `AE_HIDDEN_UNITS`   
* Next we need to process the labels, for the same you need to execute `AE_feature_extraction/ae_merge_all_labels.py". This only needs to be executed once
* The data and labels are stored in `Dataset/AE_formed_data`. Note that whenever `AE_feature_extraction/ae_all_datasets_with_hyper_mae.py" is executed the contents in the directory will get replaced
* If you have changed the Bottleneck dimension, then you got to change the key `num_columns` in `federated_client_server/model_params.py`. 
* Execute `federated_client_server/server.py` to execute FL code. Change the number of epochs and training rounds if necessary
