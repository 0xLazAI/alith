import unittest
from unittest.mock import MagicMock
import sys
import os
import json

# Add sdks/python to sys.path at the beginning to override installed package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock alith.inference.engines to avoid import errors
sys.modules["alith.inference.engines"] = MagicMock()
sys.modules["alith.inference.engines.llamacpp"] = MagicMock()

# Mock alith.lazai.client to avoid initialization during import (default args)
mock_client_module = MagicMock()
class MockSettlementData:
    def __init__(self, id, user, cost, nonce, user_signature):
        self.id = id
        self.user = user
        self.cost = cost
        self.nonce = nonce
        self.user_signature = user_signature

mock_client_module.SettlementData = MockSettlementData
sys.modules["alith.lazai.client"] = mock_client_module
sys.modules["alith.lazai"] = MagicMock()
sys.modules["alith.lazai"].client = mock_client_module

# Mock alith.lazai.request
mock_request_module = MagicMock()
mock_request_module.NONCE_HEADER = "x-nonce"
mock_request_module.USER_HEADER = "user-address"
mock_request_module.SIGNATURE_HEADER = "x-signature"
sys.modules["alith.lazai.request"] = mock_request_module

from alith.inference.settlement import TokenBillingMiddleware  # noqa: E402
from alith.inference.config import Config  # noqa: E402

class MockRequest:
    def __init__(self, path, headers):
        self.url = MagicMock()
        self.url.path = path
        self.headers = headers

class MockResponse:
    def __init__(self, status_code, body_content):
        self.status_code = status_code
        self.body_iterator = self._body_iterator(body_content)
        self.headers = {}
        self.media_type = "application/json"

    async def _body_iterator(self, content):
        yield content

class TestSettlement(unittest.IsolatedAsyncioTestCase):
    async def test_embeddings_billing(self):
        # Mock Client and Config
        mock_client = MagicMock()
        mock_config = Config(price_per_token=10)
        
        # Initialize Middleware
        app = MagicMock()
        middleware = TokenBillingMiddleware(app, client=mock_client, config=mock_config)
        
        # Mock Request
        headers = {
            "user-address": "0x123",
            "x-nonce": "12345",
            "x-signature": "0xabc"
        }
        request = MockRequest("/v1/embeddings", headers)
        
        # Mock Response from next handler
        response_data = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }
        response_content = json.dumps(response_data).encode("utf-8")
        mock_response = MockResponse(200, response_content)
        
        # Mock call_next
        async def call_next(req):
            return mock_response
            
        # Run dispatch
        # We need to patch calculate_billing to avoid actual blockchain calls or complex mocking of client
        # But wait, calculate_billing calls client.inference_settlement_fees.
        # Let's mock client.inference_settlement_fees
        mock_client.inference_settlement_fees = MagicMock()
        
        # Execute
        response = await middleware.dispatch(request, call_next)
        
        # Verify
        if not mock_client.inference_settlement_fees.called:
            print(f"Response Status: {response.status_code}")
            if hasattr(response, "body"):
                 print(f"Response Body: {response.body}")
            # Try to read body if it's a streaming response (which it is in our mock)
            # But our mock response body_iterator is a generator.
            # The middleware returns a new Response object.
            
        # Check if inference_settlement_fees was called
        self.assertTrue(mock_client.inference_settlement_fees.called)
        
        # Check arguments
        call_args = mock_client.inference_settlement_fees.call_args
        settlement_data = call_args[0][0]
        
        self.assertEqual(settlement_data.user, "0x123")
        self.assertEqual(settlement_data.cost, 5 * 10) # 5 tokens * 10 price
        self.assertTrue(isinstance(settlement_data.id, str))
        self.assertTrue(len(settlement_data.id) > 0)
        print(f"Verification Successful: Billing called for embeddings with ID: {settlement_data.id}")

if __name__ == "__main__":
    unittest.main()
