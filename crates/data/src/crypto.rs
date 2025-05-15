use openpgp::{
    Cert, Result,
    crypto::{Password, SessionKey},
    packet::{PKESK, SKESK},
    parse::{
        Parse,
        stream::{DecryptionHelper, DecryptorBuilder, MessageStructure, VerificationHelper},
    },
    policy::StandardPolicy,
    serialize::stream::{Armorer, Encryptor, LiteralWriter, Message},
    types::SymmetricAlgorithm,
};
use sequoia_openpgp as openpgp;
use std::io::{Read, Write};

/// Encrypts data using PGP encryption with a password.
///
/// This function creates a new PGP message, arms it, and encrypts the provided data
/// using the specified password. The encryption uses AES256 symmetric algorithm.
///
/// # Arguments
/// * `data` - The data to be encrypted
/// * `signature` - The password used for encryption
///
/// # Returns
/// A Result containing the encrypted data as a Vec<u8>
pub fn encrypt<S: AsRef<[u8]>>(data: S, signature: String) -> Result<Vec<u8>> {
    let mut sink = Vec::new();
    let message = Message::new(&mut sink);
    let message = Armorer::new(message).build()?;
    let password = Password::from(signature);
    let message = Encryptor::with_passwords(message, Some(password))
        .symmetric_algo(SymmetricAlgorithm::AES256)
        .build()?;
    let mut w = LiteralWriter::new(message).build()?;
    w.write_all(data.as_ref())?;
    w.finalize()?;
    Ok(sink)
}

/// Decrypts PGP-encrypted data using a password.
///
/// This function uses a custom helper to decrypt the data. It verifies the message
/// structure and decrypts it using the provided password.
///
/// # Arguments
/// * `data` - The encrypted data to be decrypted
/// * `signature` - The password used for decryption
///
/// # Returns
/// A Result containing the decrypted data as a Vec<u8>
pub fn decrypt<S: AsRef<[u8]>>(data: S, signature: String) -> Result<Vec<u8>> {
    let h = Helper { signature };
    let p = &StandardPolicy::new();
    let mut v = DecryptorBuilder::from_bytes(data.as_ref())?.with_policy(p, None, h)?;
    let mut content = Vec::new();
    v.read_to_end(&mut content)?;
    Ok(content)
}

// This fetches keys and computes the validity of the verification.
struct Helper {
    signature: String,
}

impl VerificationHelper for Helper {
    fn get_certs(&mut self, _ids: &[openpgp::KeyHandle]) -> Result<Vec<Cert>> {
        // Feed the Certs to the verifier here
        Ok(Vec::new())
    }
    fn check(&mut self, _structure: MessageStructure) -> Result<()> {
        // Implement the verification policy here.
        Ok(())
    }
}

impl DecryptionHelper for Helper {
    fn decrypt(
        &mut self,
        _: &[PKESK],
        skesks: &[SKESK],
        _sym_algo: Option<SymmetricAlgorithm>,
        decrypt: &mut dyn FnMut(Option<SymmetricAlgorithm>, &SessionKey) -> bool,
    ) -> Result<Option<Cert>> {
        let password = Password::from(self.signature.clone());
        skesks[0]
            .decrypt(&password)
            .map(|(algo, session_key)| decrypt(algo, &session_key))?;
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use crate::crypto::{decrypt, encrypt};
    use anyhow::Result;

    #[test]
    fn test_crypto() -> Result<()> {
        let data = b"Hello, PGP!";
        let password = "strong_password";
        let encrypted = encrypt(data, password.to_string())?;
        let decrypted = decrypt(&encrypted, password.to_string())?;
        assert_eq!(decrypted.as_slice(), data);
        Ok(())
    }
}
