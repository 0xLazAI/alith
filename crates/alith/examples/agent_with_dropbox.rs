use alith::data::storage::{DataStorage, DropboxStorage, UploadOptions};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let data = b"Your Data";
    let name = "file.txt";
    let token = std::env::var("DROPBOX_API_TOKEN")?;
    let storage = DropboxStorage::default();
    let file_meta = storage
        .upload(
            UploadOptions::builder()
                .data(data.to_vec())
                .name(name.to_string())
                .token(token.clone())
                .build(),
        )
        .await?;
    println!("Upload file to the dropbox: {:?}", file_meta);
    println!(
        "Get the shared link: {:?}",
        storage.get_share_link(token, file_meta.id).await?
    );
    Ok(())
}
