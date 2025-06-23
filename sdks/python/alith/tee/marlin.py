"""Alith Marlin TEE Integration & SDK. This SDK provides a Rust client for communicating with the attestation server.

For local development and testing without TDX devices, you can use the simulator available for download here:
https://github.com/marlinprotocol/oyster-monorepo/tree/master/attestation/server-custom-mock and then set the
environment variable `MARLIN_ATTESTATION_ENDPOINT` (Optional, default is http://127.0.0.1:1350)

# From Source
```no_check
git clone https://github.com/marlinprotocol/oyster-monorepo
cd oyster-monorepo/attestation/server-custom-mock

# Listens on 127.0.0.1:1350 by default
cargo run -r

# To customize listening interface and port
cargo run -r --ip-addr <ip>:<port>
```
# From Docker
```no_check
# The server runs on 1350 inside Docker, can remap to any interface and port
docker run --init -p http://127.0.0.1:1350 marlinorg/attestation-server-custom-mock
```
"""

import os
import requests
from typing import Optional

DEFAULT_MARLIN_ATTESTATION_ENDPOINT = "http://127.0.0.1:1350"
MARLIN_ATTESTATION_ENDPOINT_ENV = "MARLIN_ATTESTATION_ENDPOINT"


class MarlinError(Exception):
    """General error class for Marlin client operations"""

    def __init__(self, message):
        super().__init__(message)


class AttestationRequest:
    """Structure for attestation requests, containing public key, user data, and nonce"""

    def __init__(
        self,
        public_key: Optional[bytes] = None,
        user_data: Optional[bytes] = None,
        nonce: Optional[bytes] = None,
    ):
        self.public_key = public_key
        self.user_data = user_data
        self.nonce = nonce


class MarlinClient:
    """Main Marlin client class for managing connections to attestation services"""

    def __init__(self, endpoint: Optional[str] = None):
        self.client = requests.Session()
        self.endpoint = self._get_endpoint(endpoint)

    def _get_endpoint(self, endpoint: Optional[str]) -> str:
        """Determine the endpoint URL, prioritizing provided value, env variable, then default"""
        return endpoint or os.getenv(
            MARLIN_ATTESTATION_ENDPOINT_ENV, DEFAULT_MARLIN_ATTESTATION_ENDPOINT
        )

    def attestation_hex(self, req: AttestationRequest) -> str:
        """
        Generate a remote attestation and return hex-encoded result

        :param req: Attestation request containing public key, user data, and nonce
        :return: Hex-encoded attestation response
        :raises MarlinError: If HTTP request fails or response processing errors occur
        """
        try:
            # Convert bytes to hex strings, handling None values
            public_key_hex = self._to_hex(req.public_key)
            user_data_hex = self._to_hex(req.user_data)
            nonce_hex = self._to_hex(req.nonce)

            # Construct request URL with query parameters
            url = f"{self.endpoint}/attestation/hex"
            params = {
                "public_key": public_key_hex,
                "user_data": user_data_hex,
                "nonce": nonce_hex,
            }

            # Send GET request and process response
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            raise MarlinError(f"HTTP request error: {str(e)}")
        except Exception as e:
            raise MarlinError(f"Response processing error: {str(e)}")

    def _to_hex(self, data: Optional[bytes]) -> str:
        """Convert bytes to hex string, return empty string if None"""
        return data.hex() if data else ""

    @classmethod
    def default(cls) -> "MarlinClient":
        """Create a default Marlin client instance with auto-detected endpoint"""
        return cls()
