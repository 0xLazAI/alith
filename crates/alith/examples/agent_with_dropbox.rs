use alith::data::storage::{DropboxUploader, FileUploader, UploadOptions};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let data = b"Your Data";
    let name = "file.txt";
    let token = std::env::var("DROPBOX_API_TOKEN")?;
    let uploader = DropboxUploader::default();
    let file_meta = uploader
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
        uploader.get_share_link(token, file_meta.id).await?
    );
    Ok(())
}
