pub use aes_gcm::{Aes256Gcm, KeyInit, aead::Aead};
use anyhow::Result;
use rand_core::RngCore;

use x25519_dalek::PublicKey;
use x25519_dalek::StaticSecret;

#[inline]
fn u8_array_to_hex(buffer: &[u8]) -> String {
    buffer.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Encrypts environment variables.
///
/// # Arguments
/// * `envs` - A list of environment variables to be encrypted.
/// * `remote_pubkey` - The public key of the remote server.
///
/// # Returns
/// Returns a hexadecimal string containing the encrypted data.
pub fn encrypt_env_vars(envs: &[(&str, &str)], remote_pubkey: [u8; 32]) -> Result<String> {
    let envs_json = serde_json::to_string(&envs)?;
    let private_key = StaticSecret::random();
    let public_key = PublicKey::from(&private_key);
    let shared = private_key.diffie_hellman(&PublicKey::from(remote_pubkey));
    let encryption_key = shared.as_bytes();
    let mut iv = [0u8; 12];
    rand::rng().fill_bytes(&mut iv);

    let cipher = Aes256Gcm::new_from_slice(encryption_key).map_err(|err| anyhow::anyhow!(err))?;
    let plaintext = envs_json.as_bytes();
    let ciphertext = cipher
        .encrypt(&iv.into(), plaintext)
        .map_err(|err| anyhow::anyhow!(err))?;

    let mut result = Vec::new();
    result.extend_from_slice(public_key.as_bytes());
    result.extend_from_slice(&iv);
    result.extend_from_slice(&ciphertext);

    Ok(u8_array_to_hex(&result))
}
