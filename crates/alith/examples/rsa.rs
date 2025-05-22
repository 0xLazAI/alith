use alith::data::crypto::{
    EncodePrivateKey, EncodePublicKey, LineEnding, RsaPrivateKey, RsaPublicKey,
};

fn main() -> Result<(), anyhow::Error> {
    let mut rng = rand_08::thread_rng();
    let priv_key = RsaPrivateKey::new(&mut rng, 3072)?;
    let pub_key = RsaPublicKey::from(&priv_key);
    let private_pem = priv_key.to_pkcs8_pem(LineEnding::LF)?;
    println!("Private Key (PKCS#8):\n{}", private_pem.as_str());
    let public_pem = pub_key.to_public_key_pem(LineEnding::LF)?;
    println!("Public Key (X.509/SPKI):\n{}", public_pem);
    Ok(())
}
