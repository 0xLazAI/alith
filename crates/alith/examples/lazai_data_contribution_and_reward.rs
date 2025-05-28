use alith::data::crypto::{DecodeRsaPublicKey, Pkcs1v15Encrypt, RsaPublicKey, encrypt};
use alith::data::storage::{DataStorage, PinataIPFS, UploadOptions};
use alith::lazai::{Client, ProofRequest, U256};
use reqwest;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::new_default()?;
    let ipfs = PinataIPFS::default();
    // 1. Prepare your privacy data and encrypt it
    let data_file_name = "your_encrypted_data.txt";
    let privacy_data = "Your Privacy Data";
    let encryption_seed = "Sign to retrieve your encryption key";
    let password = client
        .wallet
        .sign_message_hex(encryption_seed.as_bytes())
        .await?;
    let encrypted_data = encrypt(privacy_data, password.clone())?;
    // 2. Upload the privacy data to IPFS and get the shared url
    let token = std::env::var("IPFS_JWT")?;
    let file_meta = ipfs
        .upload(
            UploadOptions::builder()
                .data(encrypted_data)
                .name(data_file_name.to_string())
                .token(token.clone())
                .build(),
        )
        .await?;
    let url = ipfs.get_share_link(token, file_meta.id).await?;
    // 3. Upload the privacy url to LazAI
    let mut file_id = client.get_file_id_by_url(url.as_str()).await?;
    if file_id.is_zero() {
        file_id = client.add_file(url.as_str()).await?;
    }
    // 4. Request proof in the verified computing node
    client.request_proof(file_id, U256::from(100)).await?;
    let job_id = client.file_job_ids(file_id).await?.last().cloned().unwrap();
    let job = client.get_job(job_id).await?;
    let node_info = client.get_node(job.nodeAddress).await?.unwrap();
    let node_url = node_info.url;
    let pub_key = node_info.publicKey;
    let pub_key = RsaPublicKey::from_pkcs1_pem(&pub_key)?;
    let mut rng = rand_08::thread_rng();
    let encryption_key = pub_key.encrypt(&mut rng, Pkcs1v15Encrypt, password.as_bytes())?;
    let encryption_key = hex::encode(encryption_key);
    let response = reqwest::Client::new()
        .post(format!("{node_url}/proof"))
        .json(
            &ProofRequest::builder()
                .job_id(job_id.to())
                .file_id(file_id.to())
                .file_url(url)
                .encryption_key(encryption_key)
                .encryption_seed(encryption_seed.to_string())
                .build(),
        )
        .send()
        .await?;
    if response.status().is_success() {
        println!("Proof request sent successfully")
    } else {
        println!("Failed to send proof request: {:?}", response)
    }
    // 5. Request DAT reward
    client.request_reward(file_id, None).await?;
    println!("Reward requested for file id {}", file_id);
    Ok(())
}
