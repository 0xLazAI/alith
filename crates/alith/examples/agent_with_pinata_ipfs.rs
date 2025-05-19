use alith::data::storage::{FileUploader, PinataIPFS, UploadOptions};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let data = b"Your Data";
    let name = "file.txt";
    let token = std::env::var("IPFS_JWT")?;
    let ipfs = PinataIPFS::default();
    let file_meta = ipfs
        .upload(
            UploadOptions::builder()
                .data(data.to_vec())
                .name(name.to_string())
                .token(token.clone())
                .build(),
        )
        .await?;
    println!("Upload file to the Pinata IPFS: {:?}", file_meta);
    println!(
        "Get the shared link: {:?}",
        ipfs.get_share_link(token, file_meta.id).await?
    );
    Ok(())
}
