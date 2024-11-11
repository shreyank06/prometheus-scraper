import json
import os
import statistics
import re

class JSONProcessor:
    def __init__(self, predictions_dir, random_error):
        self.predictions_dir = predictions_dir
        self.random_error = random_error

    def process_json(self, component):
        json_filename = os.path.join(self.predictions_dir, f"{component}_predictions.json")

        if not os.path.exists(json_filename):
            raise FileNotFoundError(f"{json_filename} does not exist.")

        with open(json_filename, 'r') as json_file:
            data = json.load(json_file)

        # Gather all unique subkeys from input and output
        unique_subkeys = set()

        # Process inputs and gather subkeys
        for key in data.get("inputs", {}):
            if "cm" in key and self._is_packet_session_transaction_or_global(key, component):
                percentage_deviation = self._calculate_standard_deviation(data["inputs"], key)
                if percentage_deviation is not None and percentage_deviation < 2:
                    unique_subkeys.update(self._generate_subkeys(key, component))

        # Process outputs and gather subkeys
        for key in data.get("outputs", {}):
            if "cm" in key and self._is_packet_session_transaction_or_global(key, component):
                percentage_deviation = self._calculate_standard_deviation(data["outputs"], key)
                if percentage_deviation is not None and percentage_deviation < 15:
                    unique_subkeys.update(self._generate_subkeys(key, component))
        # Analyze the memory needed for each unique subkey
        return(self._analyze_memory(data, unique_subkeys, component))

    def _analyze_memory(self, data, unique_subkeys, component):
        last_subscriber_value = self._get_last_subscriber_value_from_inputs(data)

        component_log_data = {
            "component": component,
            "logs": []
        }

        # List to store all the ues_allocated_by_memory values
        ues_allocated_by_memory_list = []

            # Get the subscriber count from inputs for expected new user calculation
        subscriber_count_predicted = data["outputs"].get("subscriber_count_Connected", [])
        subscriber_count_connected = data["inputs"].get("subscriber_count_Connected", [])
        if subscriber_count_predicted:
            differences = [round(subscriber_count_predicted[i + 1]) - round(subscriber_count_predicted[i])
                        for i in range(len(subscriber_count_predicted) - 1)]
            ues_allocated_by_memory_list.append(subscriber_count_predicted[-1])

        for subkey in unique_subkeys:
            phoenix_key = f"phoenix_memory_cm_max_used_chunks_per_pool_count_{subkey}"
            chunk_create_key = f"chunk_create_count_minus_destroy_count_{subkey}"

            total_allocated_memory_subkey = subkey.replace("cm_", "")
            total_allocated_memory_key = f"total_allocated_memory_{total_allocated_memory_subkey}"
            total_allocated_memory_value = data["outputs"].get(total_allocated_memory_key, [])

            if total_allocated_memory_value:
                total_allocated_memory_last = total_allocated_memory_value[-1]

                # Calculate total memory needed based on phoenix and chunk create keys
                total_memory_needed_mb = self._calculate_total_memory_needed(data, phoenix_key, chunk_create_key, subkey)

                # Calculate total memory used and per UE memory
                total_memory_used_chunks = self._calculate_total_memory_used(data, phoenix_key, chunk_create_key)
                last_subscriber_value = self._get_last_subscriber_value_from_inputs(data)

                # Get chunksize from output
                chunksize_key = f"phoenix_memory_chunksize_{subkey}"
                chunksize_value = data["outputs"].get(chunksize_key, [1])
                chunksize_last = chunksize_value[-1]

                # Convert total memory used to KB and MB
                total_memory_used_kb = (total_memory_used_chunks * chunksize_last) / 1024
                total_memory_used_mb = total_memory_used_kb / 1024

                # Calculate per UE memory and convert it to KB and MB
                per_ue_memory_chunks = self._calculate_per_ue_memory(total_memory_used_chunks, last_subscriber_value)
                per_ue_memory_kb = (per_ue_memory_chunks * chunksize_last) / 1024
                per_ue_memory_mb = per_ue_memory_kb / 1024
                total_ues_predicted_with_error = last_subscriber_value + 40  # manually set value for demo
                memory_needed_to_function_in_kb = (total_ues_predicted_with_error + 10) * per_ue_memory_kb
                memory_needed_to_function_in_mb = (total_ues_predicted_with_error + 10) * per_ue_memory_mb

                # Check if 'Memory needed for' is greater than 'Total allocated memory'
                if memory_needed_to_function_in_mb > total_allocated_memory_last:
                    memory_difference_mb = memory_needed_to_function_in_mb - total_allocated_memory_last
                    #log_file.write(f"  Additional memory needed for {subkey} mempool to support predicted UEs: {memory_difference_mb:.2f} MB\n")
                    component_log_data["logs"].append(f" Additional memory needed for {subkey} mempool to support predicted UEs: {memory_difference_mb:.2f} MB")
                else:
                        ues_allocated_by_memory = total_allocated_memory_last / per_ue_memory_mb
                        #log_file.write(f"  Based on total allocated memory for {subkey} mempool, max_ues_possible {ues_allocated_by_memory:.2f} UEs.\n")
                        memory_difference_mb = total_allocated_memory_last - total_memory_used_mb
                        component_log_data["logs"].append(f" Total memory left for {subkey} mempool is {memory_difference_mb:.2f} mb, total memory allocated {total_allocated_memory_last:.2f} mb.")
                        component_log_data["logs"].append(f" Based on total allocated memory for {subkey} mempool, max_ues_possible {ues_allocated_by_memory:.2f} UEs.")

                # Only check for additional memory needed if the condition is true
                if total_allocated_memory_last < total_memory_needed_mb:
                    # Calculate additional memory needed
                    memory_difference_mb = total_memory_needed_mb - total_allocated_memory_last
                    memory_difference_kb = memory_difference_mb * 1024

                    #log_file.write(f"  Additional memory needed: {memory_difference_mb:.2f} MB ({memory_difference_kb:.2f} KB)\n")
                    component_log_data["logs"].append(f"Additional memory needed: {memory_difference_mb:.2f} MB ({memory_difference_kb:.2f} KB)")
                                        # Only check for additional memory needed if the condition is true
        # Return component log data along with the list of ues_allocated_by_memory values
        return component_log_data, ues_allocated_by_memory_list, subscriber_count_predicted[-1], (len(differences) + 1), subscriber_count_connected[-1]

    def _get_last_subscriber_value_from_inputs(self, data):
        """
        Get the last value of subscriber count from inputs.
        If not present, return a default value of 1 to avoid division by zero.
        """
        subscriber_count_connected = data["inputs"].get("subscriber_count_Connected", [])
        if subscriber_count_connected:
            return subscriber_count_connected[-1]
        return 1  # Default value if subscriber count is missing

    def _calculate_total_memory_used(self, data, phoenix_key, chunk_create_key):
        """
        Calculate the total memory used by summing the last values of the
        phoenix and chunk create keys from inputs. If the inputs are empty,
        get the values from outputs instead.
        """
        # Get values from inputs first
        phoenix_value = data["inputs"].get(phoenix_key, [])
        chunk_create_value = data["inputs"].get(chunk_create_key, [])

        # Check if the values are empty and retrieve from outputs if necessary
        if not phoenix_value:  # If phoenix_value is empty, get it from outputs
            phoenix_value = data["outputs"].get(phoenix_key, [])

        if not chunk_create_value:  # If chunk_create_value is empty, get it from outputs
            chunk_create_value = data["outputs"].get(chunk_create_key, [])

        # If both values exist, return their sum
        if phoenix_value and chunk_create_value:
            phoenix_last = phoenix_value[-1]
            chunk_create_last = chunk_create_value[-1]
            return phoenix_last + chunk_create_last

    def _calculate_per_ue_memory(self, total_memory_used, last_subscriber_value):
        """
        Calculate the per UE memory used by dividing the total memory used by the last subscriber count.
        """
        if last_subscriber_value > 0:
            return total_memory_used / last_subscriber_value
        return 0  # Avoid division by zero

    def _calculate_total_memory_needed(self, data, phoenix_key, chunk_create_key, subkey):
        """
        Calculate the total memory needed based on the phoenix and chunk create keys.
        """
        phoenix_value = data["outputs"].get(phoenix_key, [])
        chunk_create_value = data["outputs"].get(chunk_create_key, [])
        chunksize_key = f"phoenix_memory_chunksize_{subkey}"
        chunksize_value = data["outputs"].get(chunksize_key, [1])

        if phoenix_value and chunk_create_value:
            phoenix_last = phoenix_value[-1]
            chunk_create_last = chunk_create_value[-1]
            chunksize_last = chunksize_value[-1]

            total_memory_needed = (phoenix_last + chunk_create_last) * chunksize_last
            total_memory_needed_kb = total_memory_needed / 1024
            return total_memory_needed_kb / 1024  # Convert to MB
        return 0

    def _generate_subkeys(self, key, component):
        parts = key.split("_")
        if len(parts) < 3:
            return []
        subkey_type = parts[-2]
        return [f"cm_{subkey_type}_{component}"]

    def _is_packet_session_transaction_or_global(self, key, component):
        #print(key)
        return any(
            suffix in key for suffix in [f"packetP_{component}", f"sessionP_{component}", f"transactionP_{component}", f"globalP_{component}"]
        )

    def _calculate_standard_deviation(self, data, key):
        values = data.get(key, [])
        if not values or len(values) < 2:
            return None

        try:
            std_dev_value = statistics.stdev(values)
            mean_value = statistics.mean(values)
            return (std_dev_value / mean_value) * 100
        except statistics.StatisticsError:
            return None
