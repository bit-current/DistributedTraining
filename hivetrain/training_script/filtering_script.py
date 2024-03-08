import torch
import torch.distributed as dist
import hashlib
import time
import json

def init_process(rank, size, backend='gloo'):
    """Initialize the distributed environment."""
    dist.init_process_group(backend, rank=rank, world_size=size)

def timestamped_challenge(rank, master_rank=0):
    """Generate and verify a timestamped cryptographic challenge."""
    if rank == master_rank:
        # Master generates a timestamped challenge
        timestamp = str(time.time())
        challenge = hashlib.sha256(timestamp.encode()).hexdigest()
        # Broadcast the challenge to all miners
        challenge_tensor = torch.tensor([float(ord(c)) for c in challenge], dtype=torch.float32)
    else:
        challenge_tensor = torch.zeros(64, dtype=torch.float32)  # Assuming SHA-256 hash length

    dist.broadcast(challenge_tensor, src=master_rank)

    if rank != master_rank:
        # Verify the challenge
        received_challenge = ''.join([chr(int(c)) for c in challenge_tensor])
        # Output the hash in a JSON object
        print(json.dumps({"received_challenge": received_challenge}))

def main():
    rank = 0  # Set your rank
    size = 2  # Set total number of processes
    backend = 'gloo'  # or 'nccl' for GPU-based operations
    init_process(rank, size, backend)
    timestamped_challenge(rank)

if __name__ == "__main__":
    main()
