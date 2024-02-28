import unittest
from unittest.mock import patch
import threading
from orchestrator import app, OrchestratorState

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_update_node_onboarding(self):
        response = self.app.post('/update', json={'meta_rank': 0, 'trigger_error': False})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['world_size'], 1)
        self.assertEqual(data['state'], 'Onboarding')

    def test_update_node_training_state(self):
        # Mock the orchestrator state to simulate the training state
        with patch.object(OrchestratorState, 'state', 'Training'), \
             patch.object(OrchestratorState, 'update_node', return_value=(2, 0)) as mock_update_node:
            response = self.app.post('/update', json={'meta_rank': 1, 'trigger_error': False})
            self.assertEqual(response.status_code, 200)
            data = response.get_json()
            self.assertEqual(data['world_size'], 2)
            self.assertEqual(data['master_meta_rank'], 0)
            self.assertEqual(data['state'], 'Training')
            mock_update_node.assert_called_once_with(1, False)

    def test_cleanup_inactive_nodes(self):
        # Setup a new state with a short cleanup interval for testing
        orchestrator_state = OrchestratorState(cleanup_interval=1, onboarding_timeout=1, min_nodes_for_training=2)
        with patch('orchestrator.orchestrator_state', new=orchestrator_state):
            orchestrator_state.update_node(0)  # Add a node for updating
            time.sleep(2)  # Wait for the cleanup interval to pass
            with orchestrator_state.lock:
                self.assertEqual(len(orchestrator_state.nodes_metadata), 0)  # Expect the node to be cleaned up

    def test_check_onboarding_completion(self):
        # Setup a new state with a short onboarding timeout for testing
        orchestrator_state = OrchestratorState(cleanup_interval=300, onboarding_timeout=1, min_nodes_for_training=1)
        with patch('orchestrator.orchestrator_state', new=orchestrator_state):
            orchestrator_state.update_node(0)  # Simulate a node onboarding
            time.sleep(2)  # Wait for the onboarding timeout to pass
            self.assertEqual(orchestrator_state.state, 'Training')  # Expect the state to transition to Training

if __name__ == '__main__':
    unittest.main()
