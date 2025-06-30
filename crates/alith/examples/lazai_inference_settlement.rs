use alith::lazai::{Client, U256, address};
use alith::{Agent, Chat, ClientConfig, LLM};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let node = address!("0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194");
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
        client.add_user(U256::from(100_000_000)).await?;
    }
    client.deposit(U256::from(100_000_000)).await?;
    client
        .deposit_inference(node, U256::from(100_000_000))
        .await?;
    println!(
        "The inference account of user is {:?}",
        client
            .get_inference_account(client.wallet.address, node)
            .await?
            .user
    );
    let file_id = U256::from(1);
    let url = client.get_inference_node(node).await?.unwrap().url;
    let agent = Agent::new(
        "",
        LLM::openai_compatible_model_with_config(
            "",
            format!("{url}/v1").as_str(),
            "",
            ClientConfig::builder()
                .extra_headers(
                    client
                        .get_request_headers(node, Some(file_id), None)
                        .await?,
                )
                .build(),
        )?,
    );
    println!("{}", agent.prompt("What is Alith").await?);
    Ok(())
}
