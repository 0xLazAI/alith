use alith::lazai::{Client, U256};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::new_default()?;
    println!("Wallet address: {}", client.wallet.address);
    println!(
        "Balance of DAT: {}",
        client
            .get_dat_balance(client.wallet.address, U256::from(1))
            .await?
    );
    Ok(())
}
