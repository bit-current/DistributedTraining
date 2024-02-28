from transitions.extensions import GraphMachine as Machine

class Orchestrator(object):
    def on_enter_waiting(self):
        print("🕒 Waiting for workers to join. Please stand by...")

    def on_enter_running(self):
        print("🚀 Running task with joined workers. Let's do this!")

    def on_enter_error_handling(self):
        print("🛠️ Handling error, attempting to reset task. Hang tight!")

    def on_enter_updating(self):
        print("🔄 Updating reference checkpoint. Almost there!")

    def on_enter_idle(self):
        print("💤 Orchestrator is idle, awaiting manual intervention or new tasks.")

# Define the states
states = ['waiting', 'running', 'error_handling', 'updating', 'idle']

# Define the transitions between states
transitions = [
    {'trigger': 'start_task', 'source': 'waiting', 'dest': 'running'},
    {'trigger': 'complete_task', 'source': 'running', 'dest': 'updating'},
    {'trigger': 'task_error', 'source': 'running', 'dest': 'error_handling'},
    {'trigger': 'reset_task', 'source': 'error_handling', 'dest': 'waiting'},
    {'trigger': 'update_checkpoint', 'source': 'updating', 'dest': 'waiting'},
    {'trigger': 'error_unresolved', 'source': 'error_handling', 'dest': 'idle'},
    {'trigger': 'resolve_error', 'source': 'idle', 'dest': 'waiting'}  # Optional transition if manual intervention resolved the error
]

# Initialize the orchestrator with its state machine
orchestrator = Orchestrator()

# Initialize state machine
machine = Machine(model=orchestrator, states=states, transitions=transitions, initial='waiting', auto_transitions=False, ignore_invalid_triggers=True)

# Example usage
orchestrator.start_task()  # Moves from waiting to running
orchestrator.task_error()  # Moves from running to error_handling
orchestrator.reset_task()  # Attempts to reset the task, moving back to running
orchestrator.complete_task()  # Successfully completes the task, moves to updating
orchestrator.update_checkpoint()  # Updates checkpoint and returns to waiting for the next cycle
machine.get_graph().draw('orchestrator_state.png', prog='dot', format='png')