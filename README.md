

### Usage for data collection

1. Start ph_init
2. go to http://192.168.254.130:9090 and see prometheus has started or not, if yes, proceed with next steps
3. Clone this repo
4. Check time on the terminal of the system where the ph_init is started with this command
```
date
```
if the time is 2 hours behind the actual time, then edit the wrapper.sh script and replace the following line on line number 28 and 40 
from 
```
local start_time=$(date -d "-2 hour" +"%Y-%m-%dT%H:%M:%S")
```
to
```
local start_time=$(date +"%Y-%m-%dT%H:%M:%S")
```
5. Run ./wrapper.sh A B, where argument A is the number of registrations, and B is the number of frequencies
6. After all the registraions are over, the script will extract and load the csv files

### For configuring memory manually
Please refer to the readme in python_code directory on how to collect the data specific to configuring memory. After you have collected the data as mentioned in the readme and want to use the collected data for calculating the memory requirements, follow these steps
```
cd memory_configuration
python3 memory_configuration.py
```
then access the API started by above command by visiting **http://127.0.0.1:5000** on url

### for performing predictions
Before performing predictions if you want to suppress the warning and informations logs by tensorflow, run this command
```
export TF_CPP_MIN_LOG_LEVEL=2
```
This enviornment variable TensorFlow to only display error messages (hiding info and warning logs) when you start the Python session.

Now to perform predictions
```
cd Python Code
python3 main.py 2024-01-01T00:00:00 2024-01-02T00:00:00 ABC123 100
```

To establish socks proxy
run this command on host
```
ssh -D 1337 saparia@192.168.143.5
```
and change proxy settings on browser and choose 
```
socks host: localhost, port: 1337
```

### Group of queries: 
There are 2 groups for the list of the query, queries_1 and queries. queries will be used by the code to query prometheus. queries_1 are the total list of queries that can be used as per the requirement and modify the queries section accordingly.

### Component:
Component section lists the components of the 5G core system whose metrics are to be fetched from prometheus.

### Regroup: 
If you want to regroup columns with same mempools together, set this configuration to true.

### retrain_model: 
If you want to retrain the already trained models, set this to true.

### get_csv: 
If you want to just get the csv files and not perform any predictions, set this to true. As of now the code doesn't support getting the files and performing predictions in one run, so when you are performing predictions you will have to set this to false.

### window_width:
The "window_width" configuration specifies data windowing for time series forecasting:

    input_width: Number of timesteps used as input.
    label_width: Number of timesteps to predict.
    shift:  Indicates how far the label window is from the input window. A shift of 30 means the labels correspond to the 30 timesteps immediately after the input window.

These settings create sliding windows to train the model on past data and predict future values.

### models: 
In the thesis, only 2 types of model were used for training and predictions, multi step convolutional model and multi step lstm model. At one time only one of these 2 models can be used for training or predictions, if you want to use both the models for predictions, it can't be done parallely so adjust the boolean values accordingly. In the trained_models directory, only multi step convolutional and lstm model are trained and saved, if you want to use other models in the future, set the boolean values accordingly.

### PCA: 
As of now PCA for dimensionality reduction is not used, so always set this to false unless you have made signifcant changes in the code base to use PCA.

### check_corelation: 
As of now corelation between the features are not performed, so always set this to false unless you have made signifcant changes in the code base to perform corelation between the features.

### static_predictions: 
- If you have the csv files for performing predictions in **data_for_training** directory and want to peform predictions, set this to true. 
- If you want to train or perform predictions on a different dataset, you can first generate the dataset, the datasets will be stored in the raw_data and processed_data folders according to the configuraitons, from the processed_data, copy and paste the data you are interested to perform predictions on. 
- If the format of the data you are going to paste in the data_for_training folder is not the same as the datasets(CSV files) already existing the data_for_training folder, then manually change the new datasets to match the format of the already existing data in the data_for_training folder else the predictions will not be possible. 
  - If you want to train the models in future with a dataset with feature dimensions(features that are different than already existing in data_for_training directory, then you will have to change the model architecture and code base. Changing the code for different purpose depends on the kind of dataset and the goal, so scenarios for all different types of features can't covered in this documentation)

### Plot predictions

Set **plot_predictions** to true when you want to visualize the predictions. **NOTE:** when you are plotting predictions, only perform predictions for one single component at a time. When you try to plot predictions in parallel for multiple components, plot packages will show error. So while plotting predictions as an example for amf, change the component section in the configuraiton to 
```
    "component": ["amf"]
```
If you have 2 components
```
    "component": ["smf","amf"]
```
plotting will show error