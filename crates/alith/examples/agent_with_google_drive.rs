use alith::data::storage::{DataStorage, GoogleDriveStorage, UploadOptions};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let data = b"Your Data";
    let name = "file.txt";
    let storage = GoogleDriveStorage::default();
    println!(
        "Upload file to the google drive: {:?}",
        storage
            .upload(
                UploadOptions::builder()
                    .data(data.to_vec())
                    .name(name.to_string())
                    .token(std::env::var("GOOGLE_DRIVE_API_KEY")?)
                    .build()
            )
            .await?
    );
    Ok(())
}
