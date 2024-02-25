echo 'Starting miner with external ip ' $EXT_IP
pm2 start hivetrain/meta_miner.py --name hm-miner -- \
    --netuid 11 \
    --orchestrator-url http://127.0.0.1:5000 \
    --batch-size 1 \
    --epochs 1 \
    --miner-script hivetrain/miner_cpu_simple.py \
    --subtensor.chain_endpoint ws://127.0.0.1:9944 \
    --wallet.name miner \
    --wallet.hotkey default \
    --host-address 127.0.0.1 \
    --port 4000 \
    --neuron.vpermit_tao_limit 10 \
    --axon.port 4000 \
    --axon.ip $EXT_IP \
    --axon.external_port 4000 \
    --axon.external_ip $EXT_IP
