#!/bin/bash

python3 hivetrain/hiveminer.py \
    --initial_peers ${INITIAL_PEERS} \
    --batch_size ${BATCH_SIZE} \
    --save_every ${SAVE_EVERY}