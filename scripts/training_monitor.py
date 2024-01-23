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
        "/ip4/127.0.0.1/tcp/22162/p2p/12D3KooWGjNpqMSQaWudtrBseypWVRMccRiQAWzV1QoPzbki8fRs",
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
    default="s25_run_v3",
)

config = parser.parse_args()    
dht = hivemind.DHT(initial_peers=config.initial_peers, start=True, client_mode = True)
progress_tracker = hivemind.optim.progress_tracker.ProgressTracker(dht=dht, prefix=config.run_id, target_batch_size=config.global_batch_size_train, start = True)
while True:
    print(progress_tracker.global_progress)
    sleep(10)