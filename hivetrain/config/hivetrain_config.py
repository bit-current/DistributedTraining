import os
import torch
import argparse
import bittensor as bt
from loguru import logger

def add_meta_miner_args(parser):
    parser.add_argument("--orchestrator-url", type=str, help="URL of the orchestrator")
    parser.add_argument("--miner-script", type=str, default="miner_cpu.py", help="The miner script to execute for training")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per forward/backward pass")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument('--validator-urls', type=str, nargs="+", help='URLs of the validators for local testing only')
    parser.add_argument('--tcp-store-address', type=str, help='URLs of the validators for local testing only')
    parser.add_argument('--tcp-store-port', type=int, help='URLs of the validators for local testing only')

def add_torch_miner_args(parser):
    parser.add_argument('--rank', type=int, help='Rank of process/node in training run')
    parser.add_argument('--world-size', type=int, help='Number of processes/nodes in training run')
    parser.add_argument('--store-address', type=str,default="127.0.0.1", help='IP/URL of the TCPStore')#FIXME add the main from btt
    parser.add_argument('--store-port', type=int,default=4999, help='Port of the test TCPStore')#FIXME add the main from btt



def add_orchestrator_args(parser):
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host-address', type=str, default="127.0.0.1")

# def add_validator_args(parser):
#     parser.add_argument('--port', type=int, default=5000, help="Port for the validator")
#     parser.add_argument('--host-address', type=str, default="127.0.0.1", help="Host address for the validator")