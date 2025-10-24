use alith::lazai::{Client, U256};
use alith::data::crypto::{DecodeRsaPublicKey, RsaPublicKey, encrypt};
use alith::data::storage::{DataStorage, PinataIPFS, UploadOptions};
use sha2::{Digest, Sha256};

fn calculate_sha256(text: &str) -> String {
    let mut hasher = Sha256::new();
    let chunk_size = 8192; // Chunk for large file.
    for i in 0..(text.len() / chunk_size + 1) {
        let start = i * chunk_size;
        let end = std::cmp::min(start + chunk_size, text.len());
        hasher.update(&text[start..end].as_bytes());
    }

    hex::encode(hasher.finalize())
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    println!("🚀 Starting DAT minting example...");
    
    // Initialize LazAI client - use testnet like Python SDK
    let client = Client::new_testnet()?;
    println!("✅ LazAI client initialized");
    println!("📍 Wallet address: {}", client.wallet.address);
    
    // Check current block and balance
    println!("📊 Current block: {}", client.get_current_block().await?);
    println!("💰 Account balance: {}", client.get_balance(client.wallet.address).await?);
    
    // Check DAT balance
    println!("🎫 DAT balance: {}", client.get_dat_balance(client.wallet.address, U256::from(1)).await?);
    
    // Prepare privacy data
    let data_file_name = "test_privacy_data.txt";
    let privacy_data = "This is my private data for DAT minting test - Rust SDK";
    let privacy_data_sha256 = calculate_sha256(privacy_data);
    println!("🔐 Privacy data SHA256: {}", privacy_data_sha256);
    
    // Encrypt the privacy data
    let encryption_seed = "Sign to retrieve your encryption key";
    let password = client
        .wallet
        .sign_message_hex(encryption_seed.as_bytes())
        .await?;
    let encrypted_data = encrypt(privacy_data, password.clone())?;
    println!("🔒 Data encrypted successfully");
    
    // Upload to IPFS and use the actual IPFS URL for LazAI registration
    if let Ok(token) = std::env::var("IPFS_JWT") {
        println!("📤 Uploading to IPFS...");
        let ipfs = PinataIPFS::default();
        let file_meta = ipfs
            .upload(
                UploadOptions::builder()
                    .data(encrypted_data)
                    .name(data_file_name.to_string())
                    .token(token.clone())
                    .build(),
            )
            .await?;
        let ipfs_url = ipfs.get_share_link(token, file_meta.id).await?;
        println!("🔗 IPFS URL: {}", ipfs_url);
        
        // Upload the actual IPFS URL to LazAI
        println!("📤 Registering file with LazAI...");
        let mut file_id = client.get_file_id_by_url(ipfs_url.as_str()).await?;
        if file_id.is_zero() {
            println!("📝 File not found, adding new file...");
            file_id = client
                .add_file(ipfs_url.as_str())
                .await?;
        }
        println!("📁 File ID: {}", file_id);
        
        // Get public key for permissions
        println!("🔑 Retrieving public key...");
        let pub_key = client.get_public_key().await?;
        let _pub_key = RsaPublicKey::from_pkcs1_pem(&pub_key)?;
        println!("🔑 Public key retrieved successfully");
        
        // Request proof for DAT minting
        println!("🎯 Requesting proof for DAT minting...");
        match client.request_proof(file_id, U256::from(100)).await {
            Ok(_) => {
                println!("✅ Proof requested successfully!");
                println!("🎉 DAT minting process completed!");
            }
            Err(e) => {
                println!("⚠️  Proof request failed: {}", e);
                println!("💡 This might be due to testnet limitations or contract state");
                println!("🔗 Your data is successfully uploaded to IPFS: {}", ipfs_url);
            }
        }
    } else {
        println!("⚠️  IPFS_JWT environment variable not set. Skipping IPFS upload and DAT minting.");
        println!("💡 To test full DAT minting, set IPFS_JWT environment variable with your Pinata JWT token.");
        
        // Still demonstrate basic LazAI functionality
        println!("🔄 Testing basic LazAI operations...");
        
        // Test transfer (small amount)
        let to = alith::lazai::address!("0x34d9E02F9bB4E4C8836e38DF4320D4a79106F194");
        let value = U256::from(1);
        println!("💸 Testing transfer of {} wei to {}", value, to);
        // Uncomment the next line to actually perform transfer
        // client.transfer(to, value, 21000, None).await?;
        println!("✅ Transfer test completed (commented out for safety)");
    }
    
    println!("🎉 DAT minting example completed!");
    Ok(())
}