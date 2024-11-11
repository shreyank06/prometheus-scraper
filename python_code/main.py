# main.py

from data_collection.collect_data import CsvCollector
import json
from datetime import datetime
import sys
import pandas as pd
from static_prediction.split_and_predict import Predictor
import threading
import os
import queue
import random

def process_dataset(dataset_name, config, dataset_folder, result_queue, component, plot, random_error):
    # Process a single dataset by creating a Predictor object, running the split method, and adding results to the queue
    print(f"\nProcessing dataset: {dataset_name}")
    df = pd.read_csv(f"{dataset_folder}/{dataset_name}")
    
    # Convert DataFrame to numeric, handling non-numeric values as NaN
    numeric_df = df.apply(pd.to_numeric, errors='coerce', axis=1)
    numeric_df['timestamp'] = df['timestamp']  # Keep the original timestamps

    # Create a Predictor instance and execute split method
    predictor = Predictor(numeric_df, config, dataset_name, component, plot, random_error)
    result = predictor.split()

    # Add the results to the queue with the component as the key
    result_queue.put((component, result))

def main(start_time, end_time, registration_number, ops_per_second):
    # Load configuration file to get settings for data collection and predictions
    with open("../processing_configuration.json", "r") as config_file:
        config = json.load(config_file)

    components = config['component']  # Components to process
    
    # Step 1: Collect CSV data if 'get_csv' is enabled in the config
    if config.get("get_csv", False):
        if isinstance(components, list):
            # Process each component in a separate thread if multiple components exist
            threads = []
            for component in components:
                thread = threading.Thread(target=collect_data_for_component, args=(start_time, end_time, registration_number, ops_per_second, component))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to finish
            for thread in threads:
                thread.join()
        else:
            # Process single component without threading
            collect_data_for_component(start_time, end_time, registration_number, ops_per_second, components)

    # Step 2: Perform static predictions if 'static_predictions' is enabled
    if config.get("static_predictions", False):
        dataset_folder = "data_for_training"  # Folder containing datasets
        datasets = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

        result_queue = queue.Queue()  # Queue to collect results from threads
        threads = []

        # Start a thread for each dataset
        for dataset_name in datasets:
            component = dataset_name.split('_')[0]

            # Create and start a thread to process each dataset
            thread = threading.Thread(
                target=process_dataset,
                args=(dataset_name, config, dataset_folder, result_queue, component, config.get("plot_prediction"), random.randint(30, 39))
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Initialize combined logs and lists to store UE data
        combined_logs = {}
        all_ue_allocations = []  # List to store UEs from all components
        ue_prediction_logs = []  # List to store UE prediction logs for comparison

        # Retrieve results from the queue
        while not result_queue.empty():
            component, (log_data, ues_allocated_by_memory_list, total_ues_predicted, results, total_ues_attached) = result_queue.get()
            
            # Store total UEs attached and log data for each component
            combined_logs["Total UEs Attached so far"] = [f"{round(total_ues_attached)} ues"]
            combined_logs[component] = log_data

            # Accumulate UE allocations from each componentpp
            all_ue_allocations.extend(ues_allocated_by_memory_list)

            # Extract UE prediction logs for comparison
            if isinstance(log_data, dict) and "logs" in log_data:
                for log in log_data["logs"]:
                    if "In total UE count predicted based on error in next 30 seconds is" in log:
                        # Extract UE count from log message
                        ue_count = float(log.split("In total UE count predicted based on error in next 30 seconds is ")[1].split(" ")[0])
                        ue_prediction_logs.append((ue_count, component, log))
        
        # Step 1: Identify the highest UE allocation from all components
        if all_ue_allocations:
            max_ue_log_str = f"In total UE count predicted based on error in next 30 seconds is {round(max(all_ue_allocations))}"
            combined_logs[max_ue_log_str] = [f"{int(round(max(all_ue_allocations)))}"]

        # Step 2: Remove all UE prediction logs for cleanup
        for component, log_data in combined_logs.items():
            if isinstance(log_data, dict) and "logs" in log_data:
                # Filter out specific UE prediction logs from logs
                log_data["logs"] = [log for log in log_data["logs"] if "In total UE count predicted based on error in next 30 seconds is" not in log]
        
        #(all_ue_allocations)
        
        # Step 3: Determine the lowest UE allocation that the system can accommodate
        if all_ue_allocations:
            lowest_ue_number = min(all_ue_allocations) - total_ues_attached
            combined_logs["Current system can accommodate"] = [f"{round(max(all_ue_allocations))} ues"]
 
        # Step 4: Save the combined logs to a JSON file
        with open('combined_predictions.json', 'w') as json_file:
            json.dump(combined_logs, json_file, indent=4)

        print("Combined logs saved to combined_predictions.json.")

def collect_data_for_component(start_time, end_time, registration_number, ops_per_second, component):
    # Load configuration settings for data collection
    with open("../processing_configuration.json", "r") as config_file:
        config = json.load(config_file)
    
    # Set start and end time, and specific component in the config
    config['start_time'] = datetime.fromisoformat(start_time)
    config['end_time'] = datetime.fromisoformat(end_time)
    config['component'] = component

    # Instantiate CsvCollector to fetch and store CSV data
    collector = CsvCollector(config['start_time'], config['end_time'], config, registration_number, ops_per_second, component)
    collector.collect_csv_data()

if __name__ == "__main__":
    # Validate command-line arguments and run main function
    if len(sys.argv) != 5:
        print("Usage: python main.py <start_time> <end_time> <registration_number> <ops_per_second>")
        sys.exit(1)
    
    # Parse arguments for start and end time, registration number, and operations per second
    start_time = sys.argv[1]
    end_time = sys.argv[2]
    registration_number = sys.argv[3]
    ops_per_second = sys.argv[4]

    # Execute main function with provided arguments
    main(start_time, end_time, registration_number, ops_per_second)
