//! Alith Marlin TEE Integration & SDK. This SDK provides a Rust client for communicating with the attestation server.
//!
//! For local development and testing without TDX devices, you can use the simulator available for download here:
//! https://github.com/marlinprotocol/oyster-monorepo/tree/master/attestation/server-custom-mock and then set the
//! environment variable `MARLIN_ATTESTATION_ENDPOINT` (Optional, default is http://127.0.0.1:1350)
//!
//! # From Source
//! ```no_check
//! git clone https://github.com/marlinprotocol/oyster-monorepo
//! cd oyster-monorepo/attestation/server-custom-mock
//!
//! # Listens on 127.0.0.1:1350 by default
//! cargo run -r
//!
//! # To customize listening interface and port
//! cargo run -r --ip-addr <ip>:<port>
//! ```
//! # From Docker
//! ```no_check
//! # The server runs on 1350 inside Docker, can remap to any interface and port
//! docker run --init -p 127.0.0.1:1350:1350 marlinorg/attestation-server-custom-mock
//! ```
use alith::tee::marlin::{AttestationRequest, MarlinClient};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = MarlinClient::default();
    println!(
        "Generate the attestation with the hex string format: {:?}",
        client
            .attestation_hex(AttestationRequest {
                user_data: Some("test".as_bytes().to_vec()),
                ..Default::default()
            })
            .await?
    );
    Ok(())
}
