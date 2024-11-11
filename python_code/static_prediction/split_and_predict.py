from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from .window_generator import WindowGenerator
from .models import Models
import matplotlib.pyplot as plt
from .pca import PCA
# Disable eager execution for performance (commented out but useful for debugging)
# tf.compat.v1.disable_eager_execution()
import matplotlib
matplotlib.use('TkAgg')  # Switch to a GUI backend like 'TkAgg' for plot display
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logs except errors

class Predictor:
    def __init__(self, df, config, dataset_name, components, plot, random_error):
        self.df = df
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.config = config
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.scaled_df = None
        self.input_width = None
        self.label_width = None
        self.shift = None
        self.label_columns = None
        self.mae_train = []
        self.mae_val = []
        self.mae_test=[]
        self.data = None
        self.pca_train_data = None
        self.pca_val_data = None
        self.pca_test_data = None
        self.dataset_name = dataset_name
        self.components = components
        self.constant_columns = None
        self.random_error = random_error
        self.plot = plot

    def find_constant_columns(self):
        constant_columns = {}

        # Iterate over each column in the original DataFrame (self.df)
        for column in self.df.columns:
            # Get the first 100 values of the column
            first_100_values = self.df[column].head(100)

            # Check if all values in the first 100 rows are the same
            if first_100_values.nunique() == 1:
                # Add the column name and the constant value to the dictionary
                constant_value = first_100_values.iloc[0]
                constant_columns[column] = constant_value
        
        # Remove the constant columns from self.df
        self.df = self.df.drop(columns=constant_columns.keys())

        return constant_columns

    def split(self):
        # Find and remove constant columns from the data
        self.constant_columns = self.find_constant_columns()

        # Convert the timestamp column to datetime
        date_time = pd.to_datetime(self.df.pop('timestamp'), format='%Y-%m-%d %H:%M:%S')

        # Scale the rest of the data
        column_indices = {name: i for i, name in enumerate(self.df.columns)}
        self.scaled_df = pd.DataFrame(self.scaler.fit_transform(self.df), columns=self.df.columns, index=self.df.index)
        
        # Split data into train, validation, and test sets
        n = len(self.scaled_df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.9)

        # Split the scaled data
        self.train_df = self.scaled_df[:train_end]
        self.val_df = self.scaled_df[train_end:val_end]
        self.test_df = self.scaled_df[val_end:]

        # Split the corresponding timestamps
        self.train_timestamps = date_time[:train_end]
        self.val_timestamps = date_time[train_end:val_end]
        self.test_timestamps = date_time[val_end:]

        print("Number of rows in test data:", len(self.test_df))
        
        # Apply PCA if needed
        if self.config['convert_to_pca']:
            pca = PCA(2)
            self.train_df, self.val_df, self.test_df = pca.convert_to_pca(self.train_df, self.val_df, self.test_df)

        # Return the predictions along with column indices
        return self.predict(column_indices)


    def predict(self, column_indices):

        num_features = self.train_df.shape[1]    

        # Initialize a wide window generator for data processing and model input
        wide_window = WindowGenerator(
                input_width=self.config['window_width']['input_width'], label_width=self.config['window_width']['label_width'], shift=self.config['window_width']['shift'],
                train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, dataset_name=self.dataset_name, scaler = self.scaler, component=self.components, 
                constant_columns = self.constant_columns, plot = self.plot, random_error = self.random_error,
                train_timestamps = self.train_timestamps, val_timestamps = self.val_timestamps, test_timestamps = self.test_timestamps)#, label_columns=[self.config['label_columns']])#+self.config['component']])

        
        # baseline_model=Models(column_indices, wide_window)
        # baseline_model.create_baseline_model()

        if self.config['models']['linear']:
            linear_model = Models(column_indices, wide_window, num_features, self.config)
            linear_model_val_performance, linear_performance = linear_model.performance_evaluation('linear', wide_window)
            self.mae_val.extend([linear_model_val_performance])
            self.mae_test.extend([linear_performance])
        if self.config['models']['densed']:
            densed_model = Models(column_indices, wide_window, num_features, self.config)
            densed_model_val_performance, densed_performance = densed_model.performance_evaluation('densed', wide_window)
            self.mae_val.extend([densed_model_val_performance])
            self.mae_test.extend([ densed_performance])
        if self.config['models']['convolutional']:
            convolutional_model = Models(column_indices, wide_window, num_features, self.config, self.components)
            convo_model_val_performance, convo_step_model_performance, results = convolutional_model.performance_evaluation('convolutional_model', wide_window)
            #self.mae_val.extend([convo_model_val_performance])
            self.mae_test.extend([convo_step_model_performance])
        if self.config['models']['lstm']:
            lstm_model = Models(column_indices, wide_window, num_features, self.config)
            lstm_model_val_performance, lstm_step_model_performance = lstm_model.performance_evaluation('lstm_model', wide_window)
            self.mae_val.extend([lstm_model_val_performance])
            self.mae_test.extend([lstm_step_model_performance])
        if self.config['models']['multi_step_linear_single_shot']:
            single_shot_linear_model = Models(column_indices, wide_window, num_features, self.config)
            single_shot_linear_model_val_performance, single_shot_linear_model_test_performance = single_shot_linear_model.performance_evaluation('single_shot_linear', wide_window)
            self.mae_val.extend([single_shot_linear_model_val_performance])
            self.mae_test.extend([single_shot_linear_model_test_performance])
        if self.config['models']['multi_step_densed_model']:
            multi_step_densed_model = Models(column_indices, wide_window, num_features, self.config)
            multi_step_densed_model_val_performance, multi_step_densed_model_test_performance = multi_step_densed_model.performance_evaluation('multi_step_densed', wide_window)
            #self.mae_val.extend([multi_step_densed_model_val_performance])
            self.mae_test.extend([multi_step_densed_model_test_performance])

        if self.config['models']['multi_step_convolutional_model']:
            multi_step_conv_model = Models(column_indices, wide_window, num_features, self.config, self.components)
            multi_step_conv_model_train_performance, multi_step_conv_model_val_performance, results = multi_step_conv_model.performance_evaluation('multi_step_conv', wide_window)
            self.mae_train.extend([multi_step_conv_model_train_performance])
            self.mae_val.extend([multi_step_conv_model_val_performance])        

        if self.config['models']['multi_step_lstm_model']:
            multi_step_lstm_model = Models(column_indices, wide_window, num_features, self.config, self.components)
            multi_step_lstm_model_train_performance, multi_step_lstm_model_val_performance, results = multi_step_lstm_model.performance_evaluation('multi_step_lstm', wide_window)
            self.mae_train.extend([multi_step_lstm_model_train_performance])
            self.mae_val.extend([multi_step_lstm_model_val_performance])

        if self.config['models']['autoregressive_lstm']:
            autoregressive_feedback_lstm = Models(column_indices, wide_window, num_features, self.config, self.components)
            multi_step_lstm_ar_train_performance, multi_step_lstm_ar_val_performance, results = autoregressive_feedback_lstm.performance_evaluation('autoregressive_lstm', wide_window)
            self.mae_train.extend([multi_step_lstm_ar_train_performance])
            self.mae_val.extend([multi_step_lstm_ar_val_performance])

        #self.plot_mae_comparison()
        return(results)
    

    def plot_mae_comparison(self):
        # Compare MAE scores between models for training and validation sets
        train_maes = []
        val_maes = []
        model_names = set()

        for train_mae in self.mae_train:
            for model_name, values in train_mae.items():
                train_maes.append(values[1])
                model_names.add(model_name)

        for val_mae in self.mae_val:
            for model_name, values in val_mae.items():
                val_maes.append(values[1])
                model_names.add(model_name)

        model_names = list(model_names)
        x = np.arange(len(model_names))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Plot the bars for validation and test MAEs side-by-side for each model
        bars1 = ax.bar(x - width/2, val_maes, width, color='blue', label='Validation MAE')
        bars2 = ax.bar(x + width/2, train_maes, width, color='orange', label='Train MAE')

        # Add labels, title, and legend
        ax.set_xlabel('Models')
        ax.set_ylabel('MAE')
        ax.set_title('Comparison of MAE for Different Models')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend()

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


