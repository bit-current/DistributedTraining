#!/bin/bash

cd /app/neurons

python3 miner-z.py \
    --initial_peers ${INITIAL_PEERS} \
    --batch_size ${BATCH_SIZE}