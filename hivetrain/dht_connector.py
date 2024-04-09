import hivemind
import bittensor as bt
import threading
import time

class DHTManager:
    def __init__(self, address_store, bittensor_network, my_uid,my_hotkey, dht_host_address, 
        dht_tcp_port, dht_udp_port, dht_external_ip, dht_private_key, store):

        self.address_store = address_store
        self.bittensor_network = bittensor_network
        self.my_uid = my_uid
        self.dht_host_address = dht_host_address
        self.dht_tcp_port = dht_tcp_port
        self.dht_udp_port = dht_udp_port
        self.dht_external_ip = dht_external_ip
        self.dht_private_key = dht_private_key
        self.logger = bt.logging
        self.my_hotkey = my_hotkey  # Set this appropriately based on your context
        self.tested_initial_peers = []
        self.check_interval = 1200
        self.storage_successful = False
        self.store = store

    def retrieve_multi_addresses(self):
        multi_addresses = []
        for uid, hotkey in enumerate(self.bittensor_network.metagraph.hotkeys):
            if uid == self.my_uid:
                retrieved_multiaddress = None
            else:
                retrieved_multiaddress = self.address_store.retrieve_multiaddress(hotkey)
            multi_addresses.append(retrieved_multiaddress)
        return multi_addresses

    def test_multi_addresses(self, multi_addresses):
        for multiaddress in multi_addresses:
            if multiaddress is None:
                continue
            try:
                self.logger.info("Is DHT Alive")
                self.logger.info(multiaddress)
                test_dht = hivemind.DHT(initial_peers=[multiaddress], start=True)
                test_dht.shutdown()
                self.logger.info("DHT Alive")
                self.tested_initial_peers.append(multiaddress)
            except Exception as e:
                self.logger.info(f"DHT failed.")
        return self.tested_initial_peers

    def initialize_my_dht(self):
        my_dht = hivemind.DHT(
            initial_peers=self.tested_initial_peers if len(self.tested_initial_peers) > 0 else None,
            start=True,
            host_maddrs=[
                f"/ip4/{self.dht_host_address}/tcp/{self.dht_tcp_port}",
                f"/ip4/{self.dht_host_address}/udp/{self.dht_udp_port}"
            ],
            announce_maddrs=[
                f"/ip4/{self.dht_external_ip}/tcp/{self.dht_tcp_port}",
                f"/ip4/{self.dht_external_ip}/udp/{self.dht_udp_port}"
            ],
            use_ipfs=False,
            use_relay=False,
            use_auto_relay=False,
            ensure_bootstrap_success=True,
            wait_timeout=180,
            bootstrap_timeout=135,
            identity_path=self.dht_private_key,
            num_replicas=1
        )
        return my_dht

    def is_multiaddress_changed(self, new_multiaddress):
        # Retrieve the current multiaddress for my_uid
        current_multiaddress = self.address_store.retrieve_multiaddress(self.my_hotkey)
        return current_multiaddress != new_multiaddress

    def update_and_store_my_multiaddress(self, my_dht):
        new_multiaddress = str(my_dht.get_visible_maddrs()[0])
        if self.is_multiaddress_changed(new_multiaddress):
            try:
                self.address_store.store_multiaddress(self.my_hotkey, new_multiaddress)
                self.logger.info(f"Multiaddress {new_multiaddress} stored successfully.")
                self.storage_successful = True
            except Exception as e:
                self.logger.error(f"Failed to store multiaddress: {e}")
                self.storage_successful = False
        else:
            self.logger.info("No change in multiaddress. Skipping update on chain.")

    def periodic_storage_check(self):
        while True:
            time.sleep(self.check_interval)
            if not self.storage_successful:
                self.logger.info("Attempting to update and store multiaddress due to previous failure.")
                my_dht = self.initialize_my_dht()
                self.update_and_store_my_multiaddress(my_dht)
            else:
                return

    def manage_dht(self):
        multi_addresses = self.retrieve_multi_addresses()
        self.test_multi_addresses(multi_addresses)
        self.my_dht = self.initialize_my_dht()
        if self.store:
            self.update_and_store_my_multiaddress(self.my_dht)
            threading.Thread(target=self.periodic_storage_check, daemon=True).start()