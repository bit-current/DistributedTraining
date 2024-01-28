import hivemind
import argparse
from time import sleep

parser = argparse.ArgumentParser()
parser.add_argument(
    "--initial_peers",
    type=str,
    nargs="+",
    help="The addresses for the DHT",
    default=[
        "/ip4/161.97.156.125/tcp/8001/p2p/12D3KooWNrfkQ8DX2RHW4c98c8As11wMNA425WTNohijyJQdA84Y",
            "/ip4/54.205.54.19/tcp/8008/p2p/12D3KooWMY4YGYZ6JkWaCNKUeKQHAuxeQcMeoNfKHbbRXVoBaMiZ",
    ],
)
parser.add_argument(
        "--global_batch_size_train",
        type=int,
        help="The hivemind global target_batch_size",
        default=1600,
)
parser.add_argument(
    "--run_id",
    type=str,
    help="The DHT run_id",
    default="s25_run_v1_1",
)

config = parser.parse_args()    
dht = hivemind.DHT(initial_peers=config.initial_peers, start=True, client_mode = True)
progress_tracker = hivemind.optim.progress_tracker.ProgressTracker(dht=dht, prefix=config.run_id, target_batch_size=config.global_batch_size_train, start = True)
while True:
    print(progress_tracker.global_progress)
    sleep(10)