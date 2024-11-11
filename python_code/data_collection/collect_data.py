from .parse_json import fetch_and_convert_data
import os
import re
import plotly.graph_objs as go

class CsvCollector:
    def __init__(self, start_time, end_time, config, registration_number, ops_per_second, component):
        # Initialize the CsvCollector with parameters for data collection and storage
        self.start_time = start_time
        self.end_time = end_time
        self.config = config
        self.connected_col_name = "subscriber_count_Connected"
        self.registration_number = registration_number
        self.ops_per_second = ops_per_second
        self.df = None
        self.component = component
        self.raw_df = None
        self.processed_df = None

    def get_existing_series(self):
        # Retrieve series numbers from existing CSV filenames to avoid overwrites
        series_numbers = []
        component_folder = self.component + '_new_csv_files_2'
        component_folder_path = os.path.join(os.getcwd(), component_folder)
        
        # Create the component folder if it doesn't exist
        if not os.path.exists(component_folder_path):
            os.makedirs(component_folder_path)
        csv_files = [f for f in os.listdir(component_folder) if f.endswith('.csv')]
        
        # List all CSV files in the component folder
        for csv_file in csv_files:
            # Extract the series number from the filename
            series_match = re.search(r'_s(\d+)_', csv_file)
            if series_match:
                series_numbers.append(int(series_match.group(1)))
        return series_numbers, component_folder_path
        
    def fetch_and_save_csv(self):
        # Save raw and processed data CSV files, creating folders if needed
        # Create folders if data transformation and regrouping are not enabled in config

        if not self.config["transform_data"] and self.config["regroup_columns"]:
            if not os.path.exists('raw_data'):
                os.makedirs('raw_data')
            if not os.path.exists('processed_data'):
                os.makedirs('processed_data')

        # Set initial filename and counter for raw data
        csv_filename = f"{self.component}_{self.registration_number}_{self.ops_per_second}_set.csv"
        counter = 0
        while os.path.exists(os.path.join('raw_data', csv_filename)):
            counter += 1
        
        csv_filename = f"{self.component}_{self.registration_number}_{self.ops_per_second}_set_{counter}.csv"
        self.raw_df.to_csv(os.path.join('raw_data', csv_filename), index=True)
        print(f"Results CSV file saved as: {csv_filename} in 'raw_data' folder.")
        
        csv_filename = f"{self.component}_{self.registration_number}_{self.ops_per_second}_set.csv"
        counter = 0
        while os.path.exists(os.path.join('processed_data', csv_filename)):
            counter += 1
            csv_filename = f"{self.component}_{self.registration_number}_{self.ops_per_second}_set_{counter}.csv"
        self.processed_df.to_csv(os.path.join('processed_data', csv_filename), index=True)
        print(f"Results CSV file saved as: {csv_filename} in 'processed_data' folder.")
           

    def collect_csv_data(self):
            # Fetch data using `fetch_and_convert_data` and save raw and processed data
            self.processed_df, self.raw_df = fetch_and_convert_data(self.config, 'queries', self.start_time, self.end_time, self.config['step'], self.component)
            self.fetch_and_save_csv()

    def visualize_data(self, df, folder_path, file_suffix):
        index_column = df.index

        num_columns = len(df.columns)

        # Create an empty figure
        fig = go.Figure()

        # Add traces for each column
        for col in df.columns[0:]:  # Start from the second column
            fig.add_trace(go.Scatter(x=index_column, y=df[col], mode='lines', name=col))

        # Update layout
        fig.update_layout(
            title=f'All Columns vs {index_column}',
            xaxis_title=str(index_column),  # Convert DatetimeIndex to string
            yaxis_title='Values',
            autosize=False,
            width=1000,
            height=600,
            # Enable interactive legend for column filtering
            legend=dict(traceorder="normal")
        )


        # Save the plot as an HTML file
        html_file_path = os.path.join(folder_path, f"{self.registration_number}_{self.ops_per_second}_{0}{file_suffix}_all_columns_vs_index.html")
        fig.write_html(html_file_path)

        # Close the figure
        #fig.close()
