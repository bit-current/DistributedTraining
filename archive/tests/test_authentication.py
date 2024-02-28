import unittest
from unittest.mock import patch
import bittensor
from meta_miner import create_signed_message
from orchestrator import authenticate_request_with_bittensor

class TestBittensorAuthentication(unittest.TestCase):
    @patch('bittensor.wallet')
    def test_successful_authentication(self, mock_wallet):
        # Setup mock wallet behavior
        mock_wallet.return_value.sign.return_value = b'signed_message'
        mock_wallet.return_value.verify.return_value = True
        mock_wallet.return_value.ss58_address = 'test_public_address'

        # Create a signed message to simulate a request
        message, signature, public_address = create_signed_message("test_message")

        # Setup the authentication function to use the mock wallet
        with patch('flask.request') as mock_request:
            mock_request.json.return_value = {
                'message': "test_message",
                'signature': signature,
                'public_address': public_address
            }

            # Mock metagraph to include the public address
            with patch('orchestrator.metagraph.hotkeys', new_callable=property) as mock_hotkeys:
                mock_hotkeys.return_value = [public_address]

                # Decorate a dummy function to test authentication
                @authenticate_request_with_bittensor
                def dummy_function():
                    return True

                # Assert that the authentication passes and calls the dummy function
                self.assertTrue(dummy_function())

    @patch('bittensor.wallet')
    def test_failed_authentication_due_to_invalid_signature(self, mock_wallet):
        # Setup mock wallet behavior for failed verification
        mock_wallet.return_value.sign.return_value = b'signed_message'
        mock_wallet.return_value.verify.return_value = False  # Simulate verification failure
        mock_wallet.return_value.ss58_address = 'test_public_address'

        # Create a signed message to simulate a request with invalid signature
        message, signature, public_address = create_signed_message("test_message")

        # Setup the authentication function to use the mock wallet
        with patch('flask.request') as mock_request:
            mock_request.json.return_value = {
                'message': "test_message",
                'signature': signature,
                'public_address': public_address
            }

            # Mock metagraph to include the public address
            with patch('orchestrator.metagraph.hotkeys', new_callable=property) as mock_hotkeys:
                mock_hotkeys.return_value = [public_address]

                # Decorate a dummy function to test authentication
                @authenticate_request_with_bittensor
                def dummy_function():
                    return True

                # Attempt to authenticate and catch the response
                with self.assertRaises(Exception) as context:
                    dummy_function()

                # Assert that the authentication fails due to invalid signature
                self.assertIn('403', str(context.exception))

if __name__ == '__main__':
    unittest.main()
