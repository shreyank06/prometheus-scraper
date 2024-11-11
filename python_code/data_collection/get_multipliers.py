import os
import pandas as pd
import json

# Filters subscriber_count_Connected data into two groups:
# 1. Zero Subscriber Count: Rows where subscriber_count_Connected = 0.
# 2. Subscriber Count 1–200: Rows where subscriber_count_Connected ranges from 1 to 200.

# Memory Usage Calculations:
# - Static Memory: Checks if memory usage is constant for zero subscribers. If so, calculates memory in MB and KB.
# - Memory per User: Calculates per-user memory based on data filtered for subscriber_count_Connected between 1 and 200. Results are stored in MB and KB.

class GetMultipliers:
    def __init__(self, directory):
        # Initialize with the directory path and set target filenames for processing
        self.directory = directory
        self.filenames = [
            "amf_300_1_set.csv",
            "ausf_300_1_set.csv",
            "smf_300_1_set.csv",
            "udm_300_1_set.csv",
            "upf1_300_1_set.csv"
        ]
        # Store DataFrames and results in dictionaries
        self.dataframes = {}
        self.results = {}

    def read_data(self):
        # Read each file and process if it exists, store error messages otherwise
        for filename in self.filenames:
            filepath = os.path.join(self.directory, filename)
            if os.path.exists(filepath):
                self.process_file(filename, filepath)
            else:
                # Store an error message if file does not exist
                self.results[filename.split("_")[0]] = {
                    "error": f"File {filename} does not exist in the directory {self.directory}"
                }

    def process_file(self, filename, filepath):
        # Read CSV file with headers to check for required columns
        df_with_header = pd.read_csv(filepath)
        if 'subscriber_count_Connected' in df_with_header.columns:
            # Process the file further if the necessary column is present
            self.process_dataframe(filename, filepath, df_with_header)
        else:
            # Store an error message if required column is missing
            self.results[filename.split("_")[0]] = {
                "error": f"'subscriber_count_Connected' column not found in {filename}"
            }

    def process_dataframe(self, filename, filepath, df_with_header):
        # Get the index of 'subscriber_count_Connected' for easy access later
        subscriber_count_column_index = df_with_header.columns.get_loc('subscriber_count_Connected')

        # Load data without headers and convert the subscriber column to numeric
        df = pd.read_csv(filepath, header=None)
        self.dataframes[filename] = df
        df[subscriber_count_column_index] = pd.to_numeric(df[subscriber_count_column_index], errors='coerce')

        # Filter rows with subscriber count zero and between 1 and 200
        subscriber_count_zeros = df[df[subscriber_count_column_index] == 0]
        subscriber_count_200 = df[(df[subscriber_count_column_index] >= 1) & (df[subscriber_count_column_index] <= 200)]

        # Evaluate memory usage for relevant columns
        self.evaluate_columns(filename, df_with_header, subscriber_count_zeros, subscriber_count_200, subscriber_count_column_index)

    def evaluate_columns(self, filename, df_with_header, subscriber_count_zeros, subscriber_count_200, subscriber_count_column_index):
        # Extract service type from filename and initialize result structure
        service_type = filename.split("_")[0]
        suffixes = [
            f"cm_globalP_{service_type}",
            f"cm_packetP_{service_type}",
            f"cm_sessionP_{service_type}",
            f"cm_transactionP_{service_type}"
        ]
        # Initialize memory status for each suffix with None values
        status = {suffix: {"static memory used": None, "memory per UE used": None, "buffer memory": None} for suffix in suffixes}
        
        # Check for required columns and process each suffix
        for suffix in suffixes:
            phoenix_col = f"phoenix_memory_cm_max_used_chunks_per_pool_count_{suffix}"
            chunk_col = f"chunk_create_count_minus_destroy_count_{suffix}"
            phoenix_memory_chunksize_col = f"phoenix_memory_chunksize_{suffix}"

            # If required columns exist, process memory calculations
            if (
                phoenix_col in df_with_header.columns and
                chunk_col in df_with_header.columns and
                phoenix_memory_chunksize_col in df_with_header.columns
            ):
                self.process_suffix(
                    filename, 
                    df_with_header, 
                    subscriber_count_zeros, 
                    subscriber_count_200, 
                    phoenix_col, 
                    chunk_col, 
                    phoenix_memory_chunksize_col, 
                    status, 
                    suffix
                )
            else:
                # Mark as missing required columns if any column is absent
                status[suffix]["static memory used"] = "Required columns not found"

        # Save results based on filename key
        filename_key = filename.split("_")[0]
        self.results[filename_key] = status

    def process_suffix(self, filename, df_with_header, subscriber_count_zeros, subscriber_count_200, phoenix_col, chunk_col, phoenix_memory_chunksize_col, status, suffix):
        # Get column indices for the required columns
        phoenix_index = df_with_header.columns.get_loc(phoenix_col)
        chunk_index = df_with_header.columns.get_loc(chunk_col)
        chunksize_index = df_with_header.columns.get_loc(phoenix_memory_chunksize_col)

        # Convert columns to numeric data types for calculations
        phoenix_data_zeros = pd.to_numeric(subscriber_count_zeros.iloc[:, phoenix_index], errors='coerce')
        chunk_data_zeros = pd.to_numeric(subscriber_count_zeros.iloc[:, chunk_index], errors='coerce')
        chunksize_data_zeros = pd.to_numeric(subscriber_count_zeros.iloc[:, chunksize_index], errors='coerce')

        phoenix_data_200 = pd.to_numeric(subscriber_count_200.iloc[:, phoenix_index], errors='coerce')
        chunk_data_200 = pd.to_numeric(subscriber_count_200.iloc[:, chunk_index], errors='coerce')
        
        # Get the first chunksize data value
        chunksize_data_value = pd.to_numeric(df_with_header.iloc[0, chunksize_index], errors='coerce')

        # Calculate memory usage if data is available
        if not phoenix_data_zeros.empty and not chunk_data_zeros.empty and pd.notna(chunksize_data_value):
            combined_data_zeros = phoenix_data_zeros + chunk_data_zeros
            unique_values_zeros = combined_data_zeros.unique()
            
            # Check if static memory remains constant
            if len(unique_values_zeros) == 1:
                mb_value_zeros = ((unique_values_zeros[0] * chunksize_data_value) / 1024) / 1024
                kb_value_zeros = mb_value_zeros * 1024
                status[suffix]["static memory used"] = f"{mb_value_zeros}mb or {kb_value_zeros}kb"

                # Analyze memory per UE if subscriber count increases
                combined_data_200 = phoenix_data_200 + chunk_data_200
                unique_values_200 = combined_data_200.unique()

                if len(unique_values_200) == 1 and unique_values_200[0] == unique_values_zeros[0]:
                    # No increase in memory with additional users
                    status[suffix]["memory per UE used"] = "no increase in memory after static memory"
                    status[suffix]["buffer memory"] = "no increase in memory after static memory"
                else:
                    # Calculate per-user memory in MB and KB
                    last_value_max_used = phoenix_data_200.iloc[-1] * chunksize_data_value
                    last_value_create_minus_destroy = chunk_data_200.iloc[-1] * chunksize_data_value

                    divided_value_mb_max_used = ((last_value_max_used / 200) / 1024) / 1024
                    divided_value_kb_max_used = divided_value_mb_max_used * 1024
                    divided_value_mb_create_minus_destroy = ((last_value_create_minus_destroy / 200) / 1024) / 1024
                    divided_value_kb_create_minus_used = divided_value_mb_create_minus_destroy * 1024

                    status[suffix]["memory per UE used"] = f"{divided_value_mb_max_used}mb or {divided_value_kb_max_used}kb"
                    status[suffix]["buffer memory"] = f"{divided_value_mb_create_minus_destroy}mb or {divided_value_kb_create_minus_used}kb"
            else:
                # Indicate variable values for static memory
                status[suffix]["static memory used"] = "Values are not the same"
        else:
            # Mark status if no data is available
            status[suffix]["static memory used"] = "No data"
            status[suffix]["memory per UE used"] = "No data"
            status[suffix]["buffer memory"] = "No data"

    def get_dataframes(self):
        # Return all processed DataFrames
        return self.dataframes

    def save_results_to_json(self, output_path):
        # Save results to JSON file at the specified path
        with open(output_path, 'w') as json_file:
            json.dump(self.results, json_file, indent=4)

# Example usage:
if __name__ == "__main__":
    # Define directory and instantiate GetMultipliers
    parent_directory = os.path.dirname(os.getcwd())
    directory = os.path.join(parent_directory, "processed_data")
    multipliers = GetMultipliers(directory)
    
    # Read data and retrieve DataFrames
    multipliers.read_data()
    dataframes = multipliers.get_dataframes()

    # Save results to JSON file
    output_json_path = os.path.join(parent_directory, "results.json")
    multipliers.save_results_to_json(output_json_path)
