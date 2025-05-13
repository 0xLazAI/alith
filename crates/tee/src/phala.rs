//! Alith Phala TEE Integration & SDK. This SDK provides a Rust client for communicating with the Tappd server,
//! which is available inside Phala Network DStack.
//!
//! For local development without TDX devices, you can use the simulator available for download here:
//! https://github.com/Leechael/tappd-simulator/releases and then set the environment variable `DSTACK_SIMULATOR_ENDPOINT`
//!
//! Leave the endpoint parameter empty for the tappd client in production. You only need to add volumes in your
//! docker-compose file to run Confidential Virtual Machines (CVMs):
//!
//! ```yaml
//!   volumes:
//!   - /var/run/tappd.sock:/var/run/tappd.sock
//! ```

use base64::{Engine as _, engine::general_purpose};
use either::Either;
use hex_literal::hex;
use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Error as HyperError, Method, Request, http::Error as HttpError};
use hyper_util::client::legacy::Client as HyperClient;
use hyper_util::client::legacy::Error as HyperClientError;
use hyperlocal::{UnixClientExt, UnixConnector, Uri};
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha384};
use std::collections::HashMap;
use std::env;
use thiserror::Error;

const INIT_MR_HEX: [u8; 48] = hex!(
    "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
);
pub const DEFAULT_TAPPD_SOCK: &str = "/var/run/tappd.sock";
pub const DSTACK_SIMULATOR_ENDPOINT_ENV: &str = "DSTACK_SIMULATOR_ENDPOINT";

/// Supported hash algorithms for attestation reports.
///
/// Specifies how report data should be processed when generating attestation quotes.
/// The `Raw` variant uses data directly without hashing, while `Empty` indicates
/// the default algorithm (sha512) should be used.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QuoteHashAlgorithm {
    Sha256,
    Sha384,
    Sha512,
    #[serde(rename = "sha3-256")]
    Sha3_256,
    #[serde(rename = "sha3-384")]
    Sha3_384,
    #[serde(rename = "sha3-512")]
    Sha3_512,
    Keccak256,
    Keccak384,
    Keccak512,
    Raw,
    /// Empty means using the default algorithm: sha512
    #[serde(rename = "")]
    #[default]
    Empty,
}

/// Comprehensive error enumeration for Tappd client operations.
#[derive(Debug, Error)]
pub enum TappdError {
    #[error("HTTP error: {0}")]
    HttpError(#[from] reqwest::Error),
    #[error("Hyper error: {0}")]
    HyperError(#[from] HyperError),
    #[error("Hyper HTTP error: {0}")]
    HyperHttpError(#[from] HttpError),
    #[error("Hyper Client error: {0}")]
    HyperClientError(#[from] HyperClientError),
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Base64 decode error: {0}")]
    Base64Error(#[from] base64::DecodeError),
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("URL parse error: {0}")]
    URLParseError(#[from] url::ParseError),
}

/// Convenience type alias for Tappd client results.
pub type Result<T> = std::result::Result<T, TappdError>;
type UnixClient = HyperClient<UnixConnector, Full<Bytes>>;

/// Response structure for key derivation operations.
///
/// Contains the derived private key and corresponding certificate chain.
/// The private key is encoded in PEM format.
#[derive(Debug, Serialize, Deserialize)]
pub struct DeriveKeyResponse {
    pub key: String,
    pub certificate_chain: Vec<String>,
}

impl DeriveKeyResponse {
    /// Extract raw bytes from the private key PEM.
    ///
    /// Removes PEM headers and decodes the base64 content.
    ///
    /// # Arguments
    /// * `max_length`: Optional maximum length for the returned byte array
    ///
    /// # Returns
    /// Raw bytes of the private key, truncated or full based on max_length.
    pub fn to_bytes(&self, max_length: Option<usize>) -> Result<Vec<u8>> {
        let content = self
            .key
            .replace("-----BEGIN PRIVATE KEY-----", "")
            .replace("-----END PRIVATE KEY-----", "")
            .replace("\n", "");

        let decoded = general_purpose::STANDARD.decode(&content)?;
        Ok(match max_length {
            Some(len) => decoded.into_iter().take(len).collect(),
            None => decoded,
        })
    }
}

/// Response structure for TDX attestation quotes.
///
/// Contains the hardware-signed attestation report and event log.
#[derive(Debug, Serialize, Deserialize)]
pub struct TdxQuoteResponse {
    pub quote: String,
    pub event_log: String,
}

impl TdxQuoteResponse {
    /// Replay RTMR history from the event log.
    ///
    /// Processes the event log to reconstruct the final RTMR values for each index.
    ///
    /// # Returns
    /// A map of RTMR indices to their corresponding final values as hex strings.
    pub fn replay_rtmrs(&self) -> Result<HashMap<u8, String>> {
        let parsed_event_log: Vec<EventLog> = serde_json::from_str(&self.event_log)?;
        let mut rtmrs = HashMap::new();

        for idx in 0..4 {
            let history: Vec<String> = parsed_event_log
                .iter()
                .filter(|e| e.imr == idx)
                .map(|e| e.digest.clone())
                .collect();
            rtmrs.insert(idx, replay_rtmr(&history));
        }

        Ok(rtmrs)
    }
}

/// Structure representing a single event in the measurement log.
#[derive(Debug, Serialize, Deserialize)]
pub struct EventLog {
    pub imr: u8,
    pub event_type: u32,
    pub digest: String,
    pub event: String,
    pub event_payload: String,
}

/// TCB (Trusted Computing Base) information structure.
///
/// Contains critical measurements of the TEE environment.
#[derive(Debug, Serialize, Deserialize)]
pub struct TcbInfo {
    pub mrtd: String,
    pub rootfs_hash: String,
    pub rtmr0: String,
    pub rtmr1: String,
    pub rtmr2: String,
    pub rtmr3: String,
    pub event_log: Vec<EventLog>,
}

/// Response structure for Tappd instance information.
///
/// Contains comprehensive metadata about the TEE instance and its configuration.
#[derive(Debug, Serialize, Deserialize)]
pub struct TappdInfoResponse {
    pub app_id: String,
    pub instance_id: String,
    pub app_cert: String,
    pub tcb_info: TcbInfo,
    pub app_name: String,
    pub public_logs: bool,
    pub public_sysinfo: bool,
}

impl TappdInfoResponse {
    /// Validate and parse the TCB information field.
    ///
    /// Handles backward compatibility with older server versions where tcb_info
    /// was encoded as a JSON string rather than structured data.
    pub fn model_validate(value: &Value) -> Result<Self> {
        let mut obj = value.clone();
        if let Some(tcb_info) = obj.get_mut("tcb_info") {
            if let Value::String(s) = tcb_info {
                *tcb_info = serde_json::from_str(s)?;
            }
        }
        Ok(serde_json::from_value(obj)?)
    }
}

/// Main Tappd client structure.
///
/// Manages connections to Tappd services and provides methods for common operations.
pub struct DstackClient {
    client: Either<Client, UnixClient>,
    endpoint: String,
}

impl DstackClient {
    /// Create a new Tappd client instance
    ///
    /// Automatically selects connection method based on endpoint:
    /// - HTTP/HTTPS URLs: Standard network connection
    /// - Other values: Treated as Unix domain socket path
    /// - Default behavior (no endpoint specified):
    ///   1. Check environment variable `DSTACK_SIMULATOR_ENDPOINT`
    ///   2. Fall back to `/var/run/tappd.sock`
    ///
    /// # Production Note
    /// In production environments, keep endpoint as None and mount the socket file:
    /// ```yaml
    /// volumes:
    ///   - /var/run/tappd.sock:/var/run/tappd.sock
    /// ```
    pub fn new(endpoint: Option<&str>) -> Self {
        let endpoint = get_endpoint(endpoint);
        if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
            Self {
                client: Either::Left(Client::new()),
                endpoint,
            }
        } else {
            let client: UnixClient = HyperClient::unix();
            Self {
                client: Either::Right(client),
                endpoint,
            }
        }
    }

    /// Derive cryptographic keys within the TEE
    ///
    /// Generates key pairs bound to the current TEE instance with hardware-backed protection.
    /// Typical use cases:
    /// - TLS certificate generation
    /// - Data encryption key derivation
    ///
    /// # Arguments
    /// * `path`: Key identifier path (e.g., "/ssl/server")
    /// * `subject`: Certificate subject (defaults to path value)
    /// * `alt_names`: Subject Alternative Names (SAN) for certificates
    ///
    /// # Returns
    /// [DeriveKeyResponse] containing private key and certificate chain
    ///
    /// # Security Note
    /// Private keys never leave the TEE environment; only certificates are returned
    pub async fn derive_key(
        &self,
        path: Option<&str>,
        subject: Option<&str>,
        alt_names: Option<&[String]>,
    ) -> Result<DeriveKeyResponse> {
        let path = path.unwrap_or("");
        let subject = subject.unwrap_or(path);
        let mut data = json!({
            "path": path,
            "subject": subject,
        });

        if let Some(names) = alt_names {
            data["alt_names"] = json!(names);
        }

        let result =
            send_rpc_request(&self.client, &self.endpoint, "/prpc/Tappd.DeriveKey", &data).await?;
        Ok(serde_json::from_value(result)?)
    }

    /// Generate TDX attestation quote
    ///
    /// Produces a hardware-signed attestation report containing:
    /// - Measurement registers (MRSIGNER, RTMR)
    /// - Event log for auditability
    ///
    /// # Arguments
    /// * `report_data`: Custom user data (16-64 bytes) included in the attestation report
    /// * `hash_algorithm`: Specifies the hashing algorithm for report data (default: sha512)
    ///
    /// # Special Handling
    /// When using `QuoteHashAlgorithm::Raw`:
    /// - Data is used directly without hashing
    /// - Must be exactly 64 bytes (128 hex characters)
    /// - Short values are zero-padded on the right
    ///
    /// # Security Recommendation
    /// Use at least sha384 to protect the integrity of report_data
    pub async fn tdx_quote(
        &self,
        report_data: impl AsRef<[u8]>,
        hash_algorithm: QuoteHashAlgorithm,
    ) -> Result<TdxQuoteResponse> {
        if report_data.as_ref().is_empty() {
            return Err(TappdError::ValidationError(
                "report_data cannot be empty".into(),
            ));
        }

        let mut hex_data = hex::encode(report_data);
        if let QuoteHashAlgorithm::Raw = hash_algorithm {
            match hex_data.len().cmp(&128) {
                std::cmp::Ordering::Less => {
                    hex_data = format!("{:0<128}", hex_data);
                }
                std::cmp::Ordering::Greater => {
                    return Err(TappdError::ValidationError(
                        "Report data too large for raw algorithm (max 64 bytes)".to_string(),
                    ));
                }
                _ => {}
            }
        }

        let payload = json!({
            "report_data": hex_data,
            "hash_algorithm": hash_algorithm,
        });

        let result = send_rpc_request(
            &self.client,
            &self.endpoint,
            "/prpc/Tappd.TdxQuote",
            &payload,
        )
        .await?;
        Ok(serde_json::from_value(result)?)
    }

    /// Retrieve TEE instance attestation information
    ///
    /// Returns metadata including:
    /// - Application identifier and instance ID
    /// - TCB (Trusted Computing Base) information
    /// - Audit log configuration
    pub async fn info(&self) -> Result<TappdInfoResponse> {
        let result =
            send_rpc_request(&self.client, &self.endpoint, "/prpc/Tappd.Info", &json!({})).await?;
        TappdInfoResponse::model_validate(&result)
    }
}

impl Default for DstackClient {
    fn default() -> Self {
        Self::new(None)
    }
}

/// Replay RTMR (Runtime Measurement Register) history to compute the final RTMR value.
///
/// RTMR is a critical component in TEE (Trusted Execution Environment) that records the runtime integrity measurements.
/// This function reconstructs the final RTMR value by iteratively hashing historical events according to these rules:
///
/// 1. Initialization with a 48-byte zero buffer (defined in INIT_MR_HEX constant)
/// 2. For each historical event:
///    - Decode the hex string to a byte array
///    - Pad with zeros to 48 bytes if shorter
///    - Concatenate current MR value with event content and compute SHA384 hash
///
/// # Arguments
/// * `history`: List of hex-encoded event digests in chronological order
///
/// # Returns
/// The final RTMR value as a hex string
fn replay_rtmr(history: &[String]) -> String {
    let mut mr = INIT_MR_HEX.to_vec();

    for content in history {
        let mut content_bytes = hex::decode(content).unwrap_or_default();
        if content_bytes.len() < 48 {
            content_bytes.resize(48, 0);
        }

        let mut hasher = Sha384::new();
        hasher.update(&mr);
        hasher.update(&content_bytes);
        mr = hasher.finalize().to_vec();
    }

    hex::encode(mr)
}

/// For local development without TDX devices, you can use the simulator available for download here:
/// https://github.com/Leechael/tappd-simulator/releases and get the endpoint from the simulator.
///
/// Leave the endpoint parameter empty for the tappd client in production.
#[inline]
fn get_endpoint(endpoint: Option<&str>) -> String {
    endpoint.map(|s| s.to_string()).unwrap_or_else(|| {
        env::var(DSTACK_SIMULATOR_ENDPOINT_ENV).unwrap_or_else(|_| DEFAULT_TAPPD_SOCK.to_string())
    })
}

/// Send an RPC request to the Tappd service
///
/// # Arguments
/// * `path`: API endpoint path (e.g., "/prpc/Tappd.DeriveKey")
/// * `payload`: JSON-formatted request body
///
/// # Returns
/// Parsed JSON response value
///
/// # Errors
/// * Network errors with HTTP status codes
/// * JSON parsing failures when response format is unexpected
async fn send_rpc_request(
    client: &Either<Client, UnixClient>,
    base_url: &str,
    path: &str,
    payload: &Value,
) -> Result<Value> {
    match client {
        Either::Left(client) => {
            let url = Url::parse(base_url)?.join(path)?;
            let response = client
                .post(url)
                .json(payload)
                .header("Content-Type", "application/json")
                .send()
                .await?;

            let json: Value = response.json().await?;
            Ok(json)
        }
        Either::Right(client) => {
            let uri = Uri::new(base_url, path);
            let body = serde_json::to_string(payload)?;
            let req: Request<Full<Bytes>> = Request::builder()
                .method(Method::POST)
                .header("Content-Type", "application/json")
                .uri(uri)
                .body(Full::from(body))?;
            let result = client.request(req).await?;
            let req_body = result.collect().await?.to_bytes();
            let json: Value = serde_json::from_slice(&req_body)?;
            Ok(json)
        }
    }
}
