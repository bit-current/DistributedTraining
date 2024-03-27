import os
import torch
import argparse
import bittensor as bt
from loguru import logger

def add_meta_miner_args(parser):
    #parser.add_argument("--meta-miner.log-activity", type=bool, help="Display logging message every request")
    parser.add_argument("--meta-miner.orchestrator-url", type=str, help="URL of the orchestrator")
    parser.add_argument("--meta-miner.miner-script", type=str, default="miner_cpu.py", help="The miner script to execute for training")
    parser.add_argument("--miner.batch-size", type=int, default=64, help="Batch size per forward/backward pass")
    parser.add_argument("--miner.epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--miner.validator-urls', type=str, nargs="+", help='URLs of the validators for local testing only')
    parser.add_argument('--miner.tcp-store-address', type=str, help='URLs of the validators for local testing only')
    parser.add_argument('--miner.tcp-store-port', type=int, help='URLs of the validators for local testing only')
    parser.add_argument('--miner.bootstrapping_server', type=str, default="http://35.239.40.23:4999/return_dht_address", help='Bootstrapping server address')
    parser.add_argument('--flask.host_address', type=str, help='URLs of the validators for local testing only')
    parser.add_argument('--flask.host_port', type=int, help='URLs of the validators for local testing only')
    

def add_torch_miner_args(parser):
    parser.add_argument('--rank', type=int, help='Rank of process/node in training run')
    parser.add_argument('--world-size', type=int, help='Number of processes/nodes in training run')
    parser.add_argument('--store-address', type=str,default="127.0.0.1", help='IP/URL of the TCPStore')#FIXME add the main from btt
    parser.add_argument('--store-port', type=int,default=4999, help='Port of the test TCPStore')#FIXME add the main from btt
    parser.add_argument(
    "--initial_peers",
    action="append",
    help="Add a peer. Can be used multiple times to pass multiple peers.",
    nargs="*",
    default=[],
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="The largest batch size able to fit on your GPU.",
        default=1,
        const=1,
        nargs="?",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        help="Save the model every X global steps.",
        default=0,
        const=0,
        nargs="?",
    )



def add_orchestrator_args(parser):
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host-address', type=str, default="127.0.0.1")

# def add_validator_args(parser):
#     parser.add_argument('--port', type=int, default=5000, help="Port for the validator")
#     parser.add_argument('--host-address', type=str, default="127.0.0.1", help="Host address for the validator")