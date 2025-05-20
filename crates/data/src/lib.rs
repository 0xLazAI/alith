//! Alith Data Package
//! - Generate encryption key
//! - Encrypt privacy data
//! - Upload encrypted data to storage registry and get the data url
//! - Add the data url to the off/on-chain data registry.
#[cfg(feature = "crypto")]
pub mod crypto;
pub mod size;
pub mod storage;
#[cfg(feature = "wallet")]
pub mod wallet;

pub use size::format_size;
