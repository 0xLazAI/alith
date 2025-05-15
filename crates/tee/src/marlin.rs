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
//! docker run --init -p http://127.0.0.1:1350 marlinorg/attestation-server-custom-mock
//! ```

use std::env;

use reqwest::Client;
use thiserror::Error;

pub const DEFAULT_MARLIN_ATTESTATION_ENDPOINT: &str = "http://127.0.0.1:1350";
pub const MARLIN_ATTESTATION_ENDPOINT_ENV: &str = "MARLIN_ATTESTATION_ENDPOINT";

/// Comprehensive error enumeration for Marlin client operations.
#[derive(Debug, Error)]
pub enum MarlinError {
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// Convenience type alias for Marlin client results.
pub type Result<T> = std::result::Result<T, MarlinError>;

#[derive(Debug, Default)]
pub struct AttestationRequest {
    pub public_key: Option<Vec<u8>>,
    pub user_data: Option<Vec<u8>>,
    pub nonce: Option<Vec<u8>>,
}

/// Main Marlin client structure.
///
/// Manages connections to Marlin services and provides methods for common operations.
pub struct MarlinClient {
    client: Client,
    endpoint: String,
}

impl MarlinClient {
    /// Create a new Marlin client instance
    ///
    /// Automatically selects connection method based on endpoint:
    /// - HTTP/HTTPS URLs: Standard network connection
    /// - Default behavior (no endpoint specified):
    ///   1. Check environment variable `MARLIN_ATTESTATION_ENDPOINT`
    ///   2. Fall back to `http://127.0.0.1:1350`
    pub fn new<S: AsRef<str>>(endpoint: Option<S>) -> Self {
        Self {
            client: Client::new(),
            endpoint: get_endpoint(endpoint),
        }
    }

    /// Generate a remote attestation with the publick key and user data.
    pub async fn attestation_hex(&self, req: AttestationRequest) -> Result<String> {
        let url = format!(
            "{}/attestation/hex?public_key={}&user_data={}&nonce={}",
            self.endpoint,
            hex::encode(req.public_key.unwrap_or_default()),
            hex::encode(req.user_data.unwrap_or_default()),
            hex::encode(req.nonce.unwrap_or_default()),
        );
        let response = self.client.get(&url).send().await?;
        Ok(response.text().await?)
    }
}

impl Default for MarlinClient {
    fn default() -> Self {
        Self::new::<&str>(None)
    }
}

#[inline]
fn get_endpoint<S: AsRef<str>>(endpoint: Option<S>) -> String {
    endpoint.map(|s| s.as_ref().to_string()).unwrap_or_else(|| {
        env::var(MARLIN_ATTESTATION_ENDPOINT_ENV)
            .unwrap_or_else(|_| DEFAULT_MARLIN_ATTESTATION_ENDPOINT.to_string())
    })
}
