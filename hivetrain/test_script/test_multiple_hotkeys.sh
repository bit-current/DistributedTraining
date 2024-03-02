#!/bin/bash

# Check if two arguments were provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <min> <max>"
    exit 1
fi

min=$1
max=$2

# Loop from min to max
for (( i=min; i<=max; i++ ))
do
    echo "Running command with torch_local_hot$i"
    btcli w new_hotkey --wallet.name torch_local_cold --wallet.hotkey torch_local_hot$i --no_version_checking
done
