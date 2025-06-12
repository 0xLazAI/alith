use alith::lazai::{Client, U256, address};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let node = address!("0x1122330000000000000000000000000000000000");
    let client = Client::new_devnet()?;
    if client.get_inference_node(node).await?.is_none() {
        client
            .add_inference_node(node, "url", "node public key")
            .await?;
    }
    // Check user already exists
    if client
        .get_user(client.wallet.address)
        .await?
        .totalBalance
        .is_zero()
    {
        client.add_user(U256::from(100_000)).await?;
    }
    client.deposit(U256::from(200_100)).await?;
    client.deposit_inference(node, U256::from(100_000)).await?;
    println!(
        "The inference account of user is {:?}",
        client
            .get_inference_account(client.wallet.address, node)
            .await?
            .user
    );
    Ok(())
}
