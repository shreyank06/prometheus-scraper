#!/bin/bash

# Function to send request and print response using curl and jq
send_request() {
    local url=$1
    local registration_number=$2
    local ops_per_second=$3
    
    # Define the parameters for the request in JSON format
    local params='{
        "jsonrpc": "2.0",
        "method": "bt_5g.gui.execution_start",
        "params": {
            "registration_number": 0,
            "deregistration_number": 0,
            "pdu_connection_number": 0,
            "handover_number": 0,
            "registration_and_pdu_number": '"$registration_number"',
            "an_release_number": 1,
            "service_request_number": 1,
            "ops_per_second": '"$ops_per_second"',
            "NB-IoT_percentage": 0
        },
        "id": 1
    }'

    # Get current timestamp as start time in milliseconds
    local start_time=$(date -d "-2 hour" +"%Y-%m-%dT%H:%M:%S.%3N")

    # Introduce a 10-second delay
    sleep 5

    # Send POST request with curl and save response
    local response=$(curl -s -X POST -H "Content-Type: application/json-rpc" -d "$params" "$url")

    # Check if execution is completed
    if [[ "$response" == *"Execution Finished"* ]]; then
        # Get current timestamp as end time in milliseconds
        local end_time=$(date -d "-2 hour" +"%Y-%m-%dT%H:%M:%S.%3N")
    fi
0

    cd "python_code"
    # Call the Python script with start and end times as arguments
    python3 main.py "$start_time" "$end_time" "$registration_number" "$ops_per_second" 

    # Print response using jq for pretty formatting
    echo "$response" | jq
}

# Main script starts here
# Define the URL
url="http://192.168.254.120:8080/jsonrpc"

# Check if the required number of arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <registration_number> <ops_per_second>"
    exit 1
fi

# Call the function to send the request
send_request "$url" "$1" "$2"
