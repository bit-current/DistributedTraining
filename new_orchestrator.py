import torch
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import create_c10d_rendezvous_handler
from torch.distributed.elastic.utils.store import create_store
import torch.multiprocessing as mp

def run_rendezvous_server(store_url, min_nodes, max_nodes, nproc_per_node):
    rdzv_config = {
        "store_url": store_url,
        "min_nodes": min_nodes,
        "max_nodes": max_nodes,
        "nproc_per_node": nproc_per_node,
    }
    rdzv_params = RendezvousParameters(
        backend="c10d",
        endpoint="",
        run_id="example",
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        **rdzv_config,
    )

    # Register c10d backend handler
    rdzv_registry.register_rendezvous_handler("c10d", create_c10d_rendezvous_handler)
    
    # Create store
    store = create_store(rdzv_params.store_url)
    
    # Create and run the c10d rendezvous handler
    handler = rdzv_registry.get_rendezvous_handler(rdzv_params, store)
    handler.run(rdzv_params.run_id)

def main():
    store_url = "file:///tmp/torchelastic_rendezvous"
    min_nodes = 1
    max_nodes = 4
    nproc_per_node = torch.cuda.device_count()  # Assuming one process per GPU

    # Running the rendezvous server in a separate process
    rdzv_server_process = mp.Process(target=run_rendezvous_server,
                                     args=(store_url, min_nodes, max_nodes, nproc_per_node))
    rdzv_server_process.start()

    # Wait for the rendezvous server process to complete
    rdzv_server_process.join()

if __name__ == "__main__":
    main()
