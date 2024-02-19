import unittest
from unittest.mock import patch, MagicMock
import meta_miner


class TestMetaMiner(unittest.TestCase):

    @patch('meta_miner.requests.post')
    def test_update_orchestrator_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {"state": "running"}
        mock_post.return_value = mock_response

        response = meta_miner.update_orchestrator("http://localhost:5000", 0)
        self.assertEqual(response, {"state": "running"})

    @patch('meta_miner.requests.post')
    def test_update_orchestrator_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.ok = False
        mock_post.return_value = mock_response

        response = meta_miner.update_orchestrator("http://localhost:5000", 0)
        self.assertIsNone(response)

    @patch('subprocess.Popen')
    def test_start_torchrun(self, mock_popen):
        process_mock = MagicMock()
        mock_popen.return_value = process_mock

        process = meta_miner.start_torchrun(4, 'miner_script.py', 'localhost:29400', 'c10d', {})
        self.assertEqual(process, process_mock)

    @patch('psutil.Process')
    def test_cleanup_child_processes_no_children(self, mock_process):
        parent_process_mock = MagicMock()
        parent_process_mock.children.return_value = []
        mock_process.return_value = parent_process_mock

        meta_miner.cleanup_child_processes(12345)  # Using an arbitrary PID for testing

        parent_process_mock.children.assert_called_with(recursive=True)

    @patch('psutil.Process')
    def test_cleanup_child_processes_with_children(self, mock_process):
        child_mock = MagicMock()
        parent_process_mock = MagicMock()
        parent_process_mock.children.return_value = [child_mock]
        mock_process.return_value = parent_process_mock

        meta_miner.cleanup_child_processes(12345)  # Using an arbitrary PID for testing

        child_mock.terminate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
