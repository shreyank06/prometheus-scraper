import requests
import pandas as pd
import sys
from sklearn.feature_extraction import DictVectorizer

def fetch_prometheus_data(config, start, end, step):
    queries = '|'.join(config['queries'])
    query = '({__name__=~"' + queries + '"})'
    params = {
        'query': query,
        'start': start.isoformat() + 'Z',
        'end': end.isoformat() + 'Z',
        'step': step
    }

    url = config['PROMETHEUS_URL']
    response = requests.get(url, params=params)
    results = response.json()['data']['result']

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.text}")


def filtered_json(config, data):
    filtered_results = []
    for result in data["data"]["result"]:
        job = result["metric"].get("job")
        component = result["metric"].get("phnf")
        if job == "phoenix" and component == config['component']:
            filtered_results.append(result)
        elif job == "phoenix" and component == 'bt' and result['metric'].get('component')=='bt5g':
            filtered_results.append(result)
        elif job == "process" and config['component'] in result["metric"].get("groupname", ""):
            filtered_results.append(result)
    return filtered_results


def used_memory(merged_df, component):
    for memtype in ['cm_globalP', 'cm_packetP', 'cm_sessionP', 'cm_transactionP']:
        alloc_col = f'phoenix_memory_allocated_{memtype}_{component}'
        waste_col = f'phoenix_memory_wasted_{memtype}_{component}'
        if all(col in merged_df for col in [alloc_col, waste_col]):
            merged_df[f'phoenix_memory_used_{memtype}_{component}'] = merged_df.pop(alloc_col).div(1000000) - merged_df.pop(waste_col).div(1000000)
    return merged_df

def memory_per_ue(merged_df, component):
    for memtype in ['cm_globalP', 'cm_packetP', 'cm_sessionP', 'cm_transactionP']:
        used_col = f'phoenix_memory_used_{memtype}_{component}'
        subscriber_count_col = f'subscriber_count_Connected'
        if all(col in merged_df for col in [used_col, subscriber_count_col]):
            merged_df[f'memory_per_ue_{memtype}_{component}'] = merged_df[used_col] / merged_df[subscriber_count_col]
    return merged_df

def process_memory_to_mb(merged_df, component):
    # Find columns matching the pattern 'process_memory_{component}_*'
    process_memory_cols = [col for col in merged_df.columns if col.startswith(f'process_memory_{component}_')]

    # Convert values in matching columns to megabytes
    for col in process_memory_cols:
        merged_df[col] = merged_df[col] / 1000000

    return merged_df

def convert_to_dataframe(config, data, component):

    merged_df = pd.DataFrame()
    # Filter the results based on the job type and component
    filtered_results = filtered_json(config, data)
    memtypes = []

    for i, metric in enumerate(filtered_results):
        job = metric['metric'].get('job', 'unknown')
        if job == 'phoenix':
            if 'http' in str(metric):
                metric_name = metric["metric"].get("__name__", "unknown")
                column_name = f'{metric_name}_{component}'
            if 'allocated' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_allocated_{memtype}_{component}'
            if "chunksize" in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_chunksize_{memtype}_{component}'
            if "chunk_count_total" in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_chunk_count_{memtype}_{component}'
            if 'wasted' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_wasted_{memtype}_{component}'
            if 'open5G_bt_subscriber_count' in str(metric):
                subscriber_state = metric["metric"].get("subscriber_state", "unknown")
                column_name = f'subscriber_count_{subscriber_state}'
            if 'allocation_count_total' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_cm_allocation_count_total_{memtype}_{component}'
            if 'max_used_chunks_per_pool_count' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_cm_max_used_chunks_per_pool_count_{memtype}_{component}'
            if 'phoenix_memory_cm_used_chunk_count' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_cm_used_chunk_count_{memtype}_{component}'
            if 'pool_create_count' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_pool_create_count_{memtype}_{component}'
            if 'pool_destroy_count' in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'phoenix_memory_pool_destroy_count_{memtype}_{component}'
                if memtype not in memtypes:
                    memtypes.append(memtype)
        if job == 'process':
            if 'cpu' in str(metric):
                mode = metric["metric"].get("mode", "unknown")
                column_name = f'cpu_{component}_{mode}'
            if "namedprocess_namegroup_memory_bytes" in str(metric):
                memtype = metric["metric"].get("memtype", "unknown")
                column_name = f'process_memory_{component}_{memtype}'
        df = pd.DataFrame(metric["values"], columns=['timestamp', column_name])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        if not merged_df.empty:
            merged_df = pd.merge(merged_df, df, how='inner', left_index=True, right_index=True)
        else:
            merged_df = df
    merged_df = merged_df.apply(pd.to_numeric, errors='ignore')
    #merged_df = merged_df.drop(columns=config["columns_to_remove"])
    if config["regroup_columns"]:
        _, raw_df = create_memtype_df(merged_df, component)
        #print(raw_df.columns)
    if config["transform_data"]:
        processed_df = multiply_columns(merged_df, component)
        processed_df = process_memory_pool_counts(merged_df, memtypes, component)
        processed_df, reordered_df = create_memtype_df(processed_df, component)
        #print(processed_df.columns)
    return processed_df, raw_df


def create_memtype_df(merged_df, component):
    # Apply numeric conversion and multiply columns
    merged_df = merged_df.apply(pd.to_numeric, errors='ignore')
    #merged_df = multiply_columns(merged_df, component)

    # Extract columns with specified substrings
    cm_columns = [col for col in merged_df.columns if 'globalP' in col or 'packetP' in col or 'sessionP' in col or 'transactionP' in col]

    # Extract remaining columns
    other_columns = [col for col in merged_df.columns if col not in cm_columns]

    # Reorder columns in the DataFrame
    reordered_columns = []
    for suffix in ['globalP', 'packetP', 'sessionP', 'transactionP']:
        reordered_columns.extend([col for col in cm_columns if suffix in col])

    reordered_columns = other_columns + reordered_columns
    reordered_df = merged_df[reordered_columns]
    memtype_df = merged_df[reordered_columns]

    # Create a new row with the appropriate labels based on the substrings in the column names
    new_row = []
    for col in reordered_columns:
        if 'globalP' in col:
            new_row.append('globalP')
        elif 'packetP' in col:
            new_row.append('packetP')
        elif 'sessionP' in col:
            new_row.append('sessionP')
        elif 'transactionP' in col:
            new_row.append('transactionP')
        else:
            new_row.append('')

    # Insert the new row as the first row in the DataFrame
    memtype_df.columns = pd.MultiIndex.from_arrays([new_row, memtype_df.columns])
    memtype_df.index = merged_df.index

    return reordered_df, memtype_df

def multiply_columns(df, component):
    # Find all columns with 'phoenix_memory_chunksize_' or 'phoenix_memory_chunk_count_' prefix
    columns_to_multiply = [col for col in df.columns if col.startswith(f'phoenix_memory_chunksize_') or col.startswith(f'phoenix_memory_chunk_count_')]
    
    # Print the columns present in the dataframe before multiplication
    for col in columns_to_multiply:
        memtype = col.split('_')[-2]  # Extract memtype from column name
        matching_columns = [c for c in df.columns if memtype in c and (c.startswith(f'phoenix_memory_chunksize_') or c.startswith(f'phoenix_memory_chunk_count_'))]  # Find all columns with the same memtype
        
        if len(matching_columns) > 1:
            total_column_name = f'total_allocated_memory_{memtype}_{component}'
            
            # Check if all matching columns exist in the dataframe
            if all(col in df.columns for col in matching_columns):
                df[total_column_name] = df[matching_columns].prod(axis=1)/1048576  # Multiply columns with the same memtype and then divide by 1048576
                
                # Filter out columns that start with 'phoenix_memory_chunksize_' from the list of columns to drop
                columns_to_drop = [c for c in matching_columns if not c.startswith('phoenix_memory_chunksize_')]
                df.drop(columns=columns_to_drop, inplace=True)  # Remove only the 'phoenix_memory_chunk_count_' columns
            else:
                print(f"Error: Columns {matching_columns} do not exist in the dataframe.")
        
    return df


def process_memory_pool_counts(df, memtypes, component):
    for memtype in memtypes:
        create_col = f"phoenix_memory_pool_create_count_{memtype}_{component}"
        destroy_col = f"phoenix_memory_pool_destroy_count_{memtype}_{component}"
        result_col = f"chunk_create_count_minus_destroy_count_{memtype}_{component}"

        if create_col in df.columns and destroy_col in df.columns:
            df[result_col] = df[create_col] - df[destroy_col]
            df.drop(columns=[create_col, destroy_col], inplace=True)
    return df

def fetch_and_convert_data(config, query_type, start_time, end_time, step, component):
    data = fetch_prometheus_data(config, start_time, end_time, step)
    return convert_to_dataframe(config, data, component)
