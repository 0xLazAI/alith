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
docker run --init -p 127.0.0.1:1350:1350 marlinorg/attestation-server-custom-mock
```
"""

from alith.tee.marlin import MarlinClient, AttestationRequest, MarlinError

# Initialize default client
client = MarlinClient.default()

# Create attestation request with sample data
request = AttestationRequest(
    public_key=b"test_public_key_bytes",
    user_data=b"test_user_data_bytes",
    nonce=b"test_nonce_bytes",
)

# Fetch attestation result
try:
    result = client.attestation_hex(request)
    print(f"Attestation result: {result[:64]}...")  # Truncate for readability
except MarlinError as e:
    print(f"Operation failed: {str(e)}")
