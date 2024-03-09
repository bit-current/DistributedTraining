#!/bin/bash

cd /app/neurons

python3 miner-z.py --initial_peers ${INITIAL_PEERS}

# python3 miner.py \
#     --netuid ${NETUID} \
#     --wallet.name ${WALLETNAME} \
#     --wallet.hotkey ${WALLETHOTKEY} \
#     --dht.port ${DHTPORT} \
#     --dht.announce_ip ${EXTERNALIP} \
#     --axon.port ${AXONPORT} \
#     --axon.external_ip ${EXTERNALIP}

# while [ true ]
# do
#   echo "I'm dead."
#   sleep 5
# done