import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' before importing pyplot
import matplotlib.pyplot as plt
import os
import json
from .json_processor import JSONProcessor  # Import the new class
tf.get_logger().setLevel('ERROR')

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=None, val_df=None, test_df=None, dataset_name = None,
               label_columns=None, scaler=None, component = None, constant_columns = None, plot = None, random_error = None, 
               train_timestamps = None, val_timestamps = None, test_timestamps = None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.dataset_name = dataset_name
    self.scaler = scaler
    #print(label_columns)
    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    self.component = component
    self.constant_columns = constant_columns
    self.random_error = random_error
    self.plot_predictions = plot
    self.train_timestamps = train_timestamps
    self.val_timestamps = val_timestamps
    self.test_timestamps = test_timestamps

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
  
  def split_window(self, features):
      
    # Split the input features into inputs and labels
      inputs = features[:, self.input_slice, :]
      labels = features[:, self.labels_slice, :]
      #print(inputs)
      if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

      # Slicing doesn't preserve static shape information, so set the shapes
      # manually. This way the `tf.data.Datasets` are easier to inspect.
      inputs.set_shape([None, self.input_width, None])
      labels.set_shape([None, self.label_width, None])

      return inputs, labels

  def make_predictions(self, inputs, model, n):
    # Make predictions using the model
    original_predictions = model(inputs)

    # Convert inputs and predictions to NumPy arrays
    inputs_np = inputs[n].numpy()
    predictions_np = original_predictions[n].numpy()

    # Reshape inputs and predictions to 2D arrays
    input_reshaped = inputs_np.reshape(-1, len(self.column_indices))
    prediction_reshaped = predictions_np.reshape(-1, len(self.column_indices))

    # Inverse transform the scaled values to original values
    inputs = self.scaler.inverse_transform(input_reshaped)
    predictions = self.scaler.inverse_transform(prediction_reshaped)

    # Prepare the data structure for JSON
    data = {
        "inputs": {},
        "outputs": {}
    }

    # Store inputs with their column names
    for i, name in enumerate(self.column_indices.keys()):
        data["inputs"][name] = inputs[:, i].tolist()

    # Store predictions with their column names
    for i, name in enumerate(self.column_indices.keys()):
        data["outputs"][name] = predictions[:, i].tolist()

    # Add constant column values to the "outputs" dictionary
    for column_name, constant_value in self.constant_columns.items():
        # Convert constant_value to a native Python type before adding to the JSON
        data["outputs"][column_name] = [int(constant_value) if isinstance(constant_value, np.integer) 
                                        else float(constant_value)]

    # Define the directory path for saving predictions one level up
    predictions_dir = os.path.join("..", "predictions_json")

    # Check if the 'predictions' directory exists, and create it if not
    os.makedirs(predictions_dir, exist_ok=True)

    # Create the JSON file name using the self.component variable inside the 'predictions' directory
    json_filename = os.path.join(predictions_dir, f"{self.component}_predictions.json")

    # Save the data into the JSON file (overwrite if it exists)
    with open(json_filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    print(f"Predictions saved to {json_filename}")
    
    # Call the new class to process the JSON file
    processor = JSONProcessor(predictions_dir, self.random_error)
    results = processor.process_json(self.component)
    return original_predictions, predictions, results

  def plot(self, dataset='train', model=None, plot_col=None, max_subplots=2):
        # Determine which dataset to use and select the corresponding timestamps
        if dataset == 'train':
            inputs, labels = next(iter(self.train))
            timestamps = self.train_timestamps
        elif dataset == 'val':
            inputs, labels = next(iter(self.val))
            timestamps = self.val_timestamps
        elif dataset == 'test':
            inputs, labels = next(iter(self.test))
            timestamps = self.test_timestamps
        elif dataset == 'example':
            inputs, labels = self.example
            timestamps = self.train_timestamps  # Assuming example is from training data
        else:
            raise ValueError("Invalid dataset. Choose 'train', 'val', 'test', or 'example'.")

        # Ensure timestamps array has sufficient length
        if len(timestamps) < len(self.input_indices) + inputs.shape[0] - 1:
            raise ValueError("The timestamp array is not long enough for the specified dataset.")

        # Prepare plot columns and scaling
        input_reshaped = inputs.numpy().reshape(-1, len(self.column_indices))
        labels_reshaped = labels.numpy().reshape(-1, len(self.column_indices))
        original_inputs = self.scaler.inverse_transform(input_reshaped).reshape(inputs.shape)
        original_labels = self.scaler.inverse_transform(labels_reshaped).reshape(labels.shape)

        if plot_col is not None:
            if plot_col not in self.column_indices:
                raise KeyError(f"'{plot_col}' not found in column indices.")
            columns_to_plot = [plot_col]
        else:
            columns_to_plot = list(self.column_indices.keys())

        # Directory setup for saving plots
        predictions_dir = 'predictions'
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        dataset_results_dir = os.path.join(predictions_dir, self.dataset_name)
        os.makedirs(dataset_results_dir, exist_ok=True)

        if model is not None:
            original_predictions, predictions, results = self.make_predictions(inputs, model, max_subplots)
            original_predictions_reshape = original_predictions.numpy().reshape(-1, len(self.column_indices))
            original_predictions = self.scaler.inverse_transform(original_predictions_reshape).reshape(original_predictions.shape)

        if(self.plot_predictions):
            for col in columns_to_plot:
                plot_col_index = self.column_indices[col]
                plt.figure(figsize=(12, 8))
                plt.ylabel(f'{col} [original scale]')

                max_n = min(max_subplots, len(inputs))
                for n in range(max_n):
                    # Calculate starting timestamp index for this sample
                    start_idx = self.input_indices[0] + n
                    sample_timestamps = timestamps[start_idx:start_idx + self.total_window_size]

                    # Plot input, label, and prediction with actual timestamps
                    plt.plot(sample_timestamps[:len(self.input_indices)], 
                            original_inputs[n, :, plot_col_index], 
                            label='Inputs', marker='.', zorder=-10)

                    if self.label_columns:
                        label_col_index = self.label_columns_indices.get(col, None)
                    else:
                        label_col_index = plot_col_index

                    if label_col_index is None:
                        print(f"Skipping labels and predictions for {col}.")
                        continue

                    plt.scatter(sample_timestamps[self.label_start:], 
                                original_labels[n, :, label_col_index],
                                edgecolors='k', label='Labels', c='#2ca02c', s=64)

                    if model is not None:
                        plt.scatter(sample_timestamps[self.label_start:], 
                                    original_predictions[n, :, label_col_index],
                                    marker='X', edgecolors='k', label='Predictions',
                                    c='#ff7f0e', s=64)

                plt.legend()
                plt.xlabel('Timestamps')
                plt.title(f'Column: {col}')
                plt.savefig(f'{dataset_results_dir}/{col}.png')
                plt.close()
        return results


    
  def make_dataset(self, data):
    # Convert data to a dataset suitable for time series
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=40,
        shuffle=False,
        batch_size=32,)
    
    # Split window data
    ds = ds.map(self.split_window)
    return ds
  
  @property
  def train(self):
    return self.make_dataset(self.train_df)

  @property
  def val(self):
    return self.make_dataset(self.val_df)

  @property
  def test(self):
    return self.make_dataset(self.test_df)

  @property
  def example(self):
    """Get and cache an example batch of `inputs, labels` for plotting."""
    result = getattr(self, '_example', None)
    if result is None:
      # No example batch was found, so get one from the `.train` dataset
      result = next(iter(self.train))
      # And cache it for next time
      self._example = result
    return result

