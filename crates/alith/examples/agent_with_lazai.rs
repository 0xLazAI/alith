use alith::lazai::{ChainConfig, ChainManager, U256, Wallet, address};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let wallet = Wallet::from_env()?;
    let chain = ChainManager::new(ChainConfig::local(), wallet)?;
    let to = address!("0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194");
    let value = U256::from(10);
    println!("The latest block: {}", chain.get_current_block().await?);
    println!(
        "Account balance: {}",
        chain.get_balance(chain.wallet.address).await?
    );
    chain.transfer(to, value, 21000, None).await?;
    println!("Transfer value {} to {}", to, value);
    Ok(())
}
