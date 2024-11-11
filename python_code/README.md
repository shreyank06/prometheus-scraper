### data_collection/get_multipliers.py

- Filters **subscriber_count_Connected** data into two groups:

    - Zero Subscriber Count: Rows where subscriber_count_Connected = 0.
    - Subscriber Count 1–200: Rows where subscriber_count_Connected ranges from 1 to 200.

- Memory Usage Calculations:

    - **Static Memory:** Checks if memory usage is constant for zero subscribers. If so, calculates memory in MB and KB.
    - **Memory per User:** Calculates per-user memory based on data filtered for subscriber_count_Connected between 1 and 200. Results are stored in MB and KB.

- usage
    ```
    - cd python_code/data_collection
    - python3 get_multipliers.py
    ```   
    - **Note** - make sure files like self.filenames = [
            "amf_300_1_set.csv",
            "ausf_300_1_set.csv",
            "smf_300_1_set.csv",
            "udm_300_1_set.csv",
            "upf1_300_1_set.csv"
        ]

        are already existing in the **processed_data** directory where the first 5 to 6 rows represent the datapoints where no subscribers attached as these rows are used to calculate the static memory. Samples of this files are already stored in the processed_data directory. 
    - If you make some changes **to the 5G core**, it is possible that the **memory usage patterns** change and in that this script will have to be executed again and generate CSV files exactly in the same format as shown above. For this you simple need to execute wrapper.sh script with
    ```
    ./wrapper.sh 300 1
    ```

    - The result of this script will be stored in **results.json** file.

### Plotting Predictions

Predictions directory will contain the predictions of the plot. As mentioned earlier, plotting can only be done for one component at a time as of now, so when you want to plot, set the plot_predictions parameter to true in the configuration and perform predictions and the plots will be stored in this directory.
