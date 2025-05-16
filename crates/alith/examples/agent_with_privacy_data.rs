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
use alith::{
    data::{
        crypto::{Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey, decrypt, encrypt},
        wallet::LocalEthWallet,
    },
    tee::marlin::{AttestationRequest, MarlinClient},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 1. Prepare the privacy data
    let privacy_data = b"Your Privacy Data";
    // 2. Get the signature from user's wallet.
    let signature = LocalEthWallet::random()?.sign().await?;
    // 3. Generate the RSA private key and public key
    let mut rng = rand::thread_rng();
    let priv_key = RsaPrivateKey::new(&mut rng, 3072)?;
    let pub_key = RsaPublicKey::from(&priv_key);
    // 4. Encrypt the privacy data and password
    let encrypted_key = pub_key.encrypt(&mut rng, Pkcs1v15Encrypt, signature.as_bytes())?;
    let encrypted_data = encrypt(privacy_data, signature.to_string())?;
    println!("Encrypted data: {:?}", hex::encode(&encrypted_data));
    println!("Encrypted key: {:?}", hex::encode(&encrypted_key));
    // 5. Decrypt the privacy data password using the RSA private key.
    let password = priv_key.decrypt(Pkcs1v15Encrypt, &encrypted_key)?;
    // 6. Decrypt the privacy data using the password
    let decrypted_data = decrypt(&encrypted_data, String::from_utf8(password)?)?;
    assert_eq!(decrypted_data.as_slice(), privacy_data);
    // 7. Generate the proof in the TEE.
    let client = MarlinClient::default();
    println!(
        "Generate the attestation within TEE: {:?}",
        client
            .attestation_hex(AttestationRequest {
                user_data: Some(decrypted_data),
                ..Default::default()
            })
            .await?
    );
    Ok(())
}
