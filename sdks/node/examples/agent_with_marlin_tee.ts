/* Alith Marlin TEE Integration & SDK. This SDK provides a Rust client for communicating with the attestation server.
 * For local development and testing without TDX devices, you can use the simulator available for download here:
 * https://github.com/marlinprotocol/oyster-monorepo/tree/master/attestation/server-custom-mock and then set the
 * environment variable `MARLIN_ATTESTATION_ENDPOINT` (Optional, default is http://127.0.0.1:1350)
 *
 * # From Source
 * ```no_check
 * git clone https://github.com/marlinprotocol/oyster-monorepo
 * cd oyster-monorepo/attestation/server-custom-mock
 *
 * # Listens on 127.0.0.1:1350 by default
 * cargo run -r
 *
 * # To customize listening interface and port
 * cargo run -r --ip-addr <ip>:<port>
 * ```
 * # From Docker
 * ```no_check
 * # The server runs on 1350 inside Docker, can remap to any interface and port
 * docker run --init -p 127.0.0.1:1350:1350 marlinorg/attestation-server-custom-mock
 * ```
 */

import {
  AttestationRequest,
  MarlinClient,
  MarlinError,
} from "alith/tee/marlin";

async function main() {
  const client = new MarlinClient();
  const request = new AttestationRequest(
    Uint8Array.from("test_public_key_bytes"),
    Uint8Array.from("test_user_data_bytes"),
    Uint8Array.from("test_nonce_bytes")
  );
  try {
    // Fetch attestation result
    const result = await client.attestationHex(request);
    console.log("Attestation result:", result);
  } catch (error) {
    if (error instanceof MarlinError) {
      console.error(`Error type: ${error.type}`);
      console.error(`Error message: ${error.message}`);
    } else {
      console.error("Unexpected error:", error);
    }
  }
}

await main();
