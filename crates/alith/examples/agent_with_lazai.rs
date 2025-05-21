use alith::lazai::{Client, U256, address};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::new_default()?;
    let to = address!("0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194");
    let value = U256::from(10);
    println!("The latest block: {}", client.get_current_block().await?);
    println!(
        "Account balance: {}",
        client.get_balance(client.wallet.address).await?
    );
    client.transfer(to, value, 21000, None).await?;
    println!("Transfer value {} to {}", to, value);
    Ok(())
}
