#!/bin/bash

# Specify the root directory for log files
log_dir="/mnt/cpfs_m6_29e5gphu/data/user/jinhui/Projects/starVLA/results/Checkpoints/0222_QwenGR00TN1d6_epx3_randominti_multi_robot_4B/logs"

# Iterate over all log files in the specified directory
last_Folder=""
find "$log_dir" -type f -name "*.log" | while read -r log_file; do
    # Extract the last "Total success rate" value from the log file
    success_rate=$(grep "INFO     | >> Total success rate:" "$log_file" | tail -n 1)
    
    # If a match is found, print the log file path and the corresponding success rate
    if [ -n "$success_rate" ]; then
        echo "Folder: $(basename "$(dirname "$log_file")")"
        echo "File: $(basename "$log_file")"
        echo "$success_rate"
        echo
    fi
done