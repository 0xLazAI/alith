use alith::data::crypto::{
    BASE64_STANDARD, Base64Engine, Pkcs1v15Encrypt, RsaPrivateKey, RsaPublicKey,
};
use alith::lazai::{Client, Permission, U256, Wallet, address};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 1. Get the privacy data encrypted_key
    let mut rng = rand_08::thread_rng();
    let private_key = RsaPrivateKey::new(&mut rng, 3072)?;
    let public_key = RsaPublicKey::from(&private_key);
    let signature = Wallet::from_env()?.sign_hex().await?;
    let mut rng = rand_08::thread_rng();
    let encrypted_key = public_key.encrypt(&mut rng, Pkcs1v15Encrypt, signature.as_bytes())?;
    let encrypted_key = BASE64_STANDARD.encode(encrypted_key);
    // 2. Upload the privacy data url and encrypted_key
    let client = Client::new_default()?;
    let to = address!("0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194");
    let value = U256::from(1);
    let url = "https://your_privacy_data_url.txt";
    println!("The latest block: {}", client.get_current_block().await?);
    println!(
        "Account balance: {}",
        client.get_balance(client.wallet.address).await?
    );
    client.transfer(to, value, 21000, None).await?;
    println!("Transfer value {} to {}", to, value);
    let file_id = client.add_file(url).await?;
    client
        .add_permission_for_file(
            file_id,
            Permission {
                account: client.config.data_registry_address,
                key: encrypted_key,
            },
        )
        .await?;
    println!(
        "Get the privacy file information {:?}",
        client.get_file(file_id).await?.ownerAddress
    );
    Ok(())
}
