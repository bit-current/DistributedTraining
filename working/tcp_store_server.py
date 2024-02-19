# tcp_store_server.py
import sys
import logging
from torch.distributed import TCPStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(store_address, store_port):
    # Convert port to integer
    store_port = int(store_port)
    try:
        # Initialize the TCPStore
        store = TCPStore(store_address, store_port, 1, True)
        logging.info(f"TCPStore running at {store_address}:{store_port}")

        # Keep the store running indefinitely
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logging.info("Shutting down TCPStore.")
    except Exception as e:
        logging.error(f"Failed to start TCPStore: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logging.error("Usage: python tcp_store_server.py <store_address> <store_port>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
