"""For the deplopment environment, set the environment variable `DSTACK_SIMULATOR_ENDPOINT` with the
simulator: https://github.com/Leechael/tappd-simulator/releases

In production environments, mount the socket file in your docker container:
```yaml
volumes:
  - /var/run/tappd.sock:/var/run/tappd.sock
```
"""

from alith.tee.phala import TappdClient, AsyncTappdClient

# Synchronous client
client = TappdClient()
async_client = AsyncTappdClient()

# Get the information of the Base Image.
info = client.info()  # or await async_client.info()
print(info.app_id)  # Application ID
print(info.tcb_info.mrtd)  # Access TCB info directly
print(info.tcb_info.event_log[0].event)  # Access event log entries

# Derive a key with optional path and subject
key_result = client.derive_key(
    "<unique-id>"
)  # or await async_client.derive_key('<unique-id>')
print(key_result.key)  # X.509 private key in PEM format
print(key_result.certificate_chain)  # Certificate chain
key_bytes = key_result.toBytes()  # Get key as bytes

# Generate TDX quote
quote_result = client.tdx_quote(
    report_data="some-data"
)  # or await async_client.tdx_quote(report_data='some-data')
print(quote_result.quote)  # TDX quote in hex format
print(quote_result.event_log)  # Event log
rtmrs = quote_result.replay_rtmrs()  # Replay RTMRs
