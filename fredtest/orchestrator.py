from transitions.extensions import GraphMachine as Machine

class Orchestrator(object):
    def __init__(self):
        self.miners = []  # List to keep track of registered miners
    
    def register_miner(self, miner_address):
        if miner_address not in self.miners:
            self.miners.append(miner_address)
            print(f"Miner {miner_address} registered.")
        else:
            print(f"Miner {miner_address} is already registered.")
    
    def deregister_miner(self, miner_address):
        if miner_address in self.miners:
            self.miners.remove(miner_address)
            print(f"Miner {miner_address} deregistered.")
        else:
            print(f"Miner {miner_address} is not registered.")
    
    def select_miner(self):
        # Simple round-robin selection for demonstration
        # In a real scenario, you could implement more complex logic here
        if self.miners:
            return self.miners.pop(0)  # This removes the first miner and returns it
        else:
            print("No miners registered.")
            return None

    def on_enter_waiting(self):
        print("Waiting for workers to join.")
        if not self.miners:
            print("No miners available, staying in waiting.")

    def on_enter_running(self):
        print("Running task with joined workers.")
        selected_miner = self.select_miner()
        if selected_miner:
            print(f"Task assigned to miner: {selected_miner}")
            # Here you would send the task to the selected miner
        else:
            self.to_error_handling()

    def on_enter_error_handling(self):
        print("Handling error, attempting to reset task.")

    def on_enter_updating(self):
        print("Updating reference checkpoint.")

    def on_enter_idle(self):
        print("Orchestrator is idle, awaiting manual intervention or new tasks.")

# Define the states and transitions as before

states = ['waiting', 'running', 'error_handling', 'updating', 'idle']

transitions = [
    {'trigger': 'start_task', 'source': 'waiting', 'dest': 'running'},
    {'trigger': 'complete_task', 'source': 'running', 'dest': 'updating'},
    {'trigger': 'task_error', 'source': 'running', 'dest': 'error_handling'},
    {'trigger': 'reset_task', 'source': 'error_handling', 'dest': 'waiting'},
    {'trigger': 'update_checkpoint', 'source': 'updating', 'dest': 'waiting'},
    {'trigger': 'error_unresolved', 'source': 'error_handling', 'dest': 'idle'},
    {'trigger': 'resolve_error', 'source': 'idle', 'dest': 'waiting'}
]

# Initialize the orchestrator with its state machine
orchestrator = Orchestrator()

# Initialize state machine
machine = Machine(model=orchestrator, states=states, transitions=transitions, initial='waiting', auto_transitions=False, ignore_invalid_triggers=True)

# Now you can register and deregister miners dynamically
orchestrator.register_miner("http://miner1.example.com")
orchestrator.register_miner("http://miner2.example.com")

# Proceed with task workflow as before
