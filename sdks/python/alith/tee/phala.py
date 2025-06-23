"""Alith Phala TEE Integration & SDK. This SDK provides a Rust client for communicating with the Tappd server,
which is available inside Phala Network DStack.

For local development without TDX devices, you can use the simulator available for download here:
https://github.com/Leechael/tappd-simulator/releases and then set the environment variable `DSTACK_SIMULATOR_ENDPOINT`

Leave the endpoint parameter empty for the tappd client in production. You only need to add volumes in your
docker-compose file to run Confidential Virtual Machines (CVMs):

```yaml
  volumes:
  - /var/run/tappd.sock:/var/run/tappd.sock
```
"""

from dstack_sdk import (
    TappdClient,
    AsyncTappdClient,
    DeriveKeyResponse,
    TdxQuoteResponse,
)

DstackClient = TappdClient
AsyncDstackClient = AsyncTappdClient
DeriveKeyResponse = DeriveKeyResponse
TdxQuoteResponse = TdxQuoteResponse
