#!/bin/bash

cd /app/neurons

python3 hiveminer.py \
    --initial_peers ${INITIAL_PEERS} \
    --batch_size ${BATCH_SIZE} \
    --save_every ${SAVE_EVERY}