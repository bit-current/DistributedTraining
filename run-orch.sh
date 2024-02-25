pm2 start hivetrain/orchestrator_simple.py --name hm-orch -- \
    --host-address 0.0.0.0 \
    --port 5000 \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --tcp-store-port 4999 \
    --tcp-store-address 127.0.0.1
