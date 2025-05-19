use anyhow::Result;
use std::fmt::Display;

#[cfg(feature = "dropbox")]
pub mod dropbox;
#[cfg(feature = "google-drive")]
pub mod google_drive;
#[cfg(feature = "ipfs")]
pub mod ipfs;

#[cfg(feature = "dropbox")]
pub use dropbox::{DROPBOX_DEFAULT_FOLDER_ENV, DropboxStorage};
#[cfg(feature = "google-drive")]
pub use google_drive::{GOOGLE_DRIVE_DEFAULT_FOLDER_ENV, GOOGLE_DRIVE_URL, GoogleDriveStorage};
#[cfg(feature = "ipfs")]
pub use ipfs::{IPFS_API_KEY_ENV, IPFS_API_SECRET_ENV, IPFS_GATEWAY_ENV, IPFS_JWT_ENV, PinataIPFS};

#[derive(Debug, Clone, Copy)]
pub enum StorageType {
    GoogleDrive,
    Dropbox,
    IPFS,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Upload error: {0}")]
    UploadError(String),
}

#[async_trait::async_trait]
pub trait DataStorage {
    async fn upload(&self, opts: UploadOptions) -> Result<FileMetadata>;
    async fn get_share_link(&self, opts: GetShareLinkOptions) -> Result<String>;
    fn storage_type(&self) -> StorageType;
}

#[derive(Clone, bon::Builder)]
pub struct UploadOptions {
    pub name: String,
    pub data: Vec<u8>,
    pub token: String,
}

#[derive(Clone, bon::Builder)]
pub struct GetShareLinkOptions {
    pub token: String,
    pub id: String,
}

impl Display for StorageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageType::GoogleDrive => write!(f, "google-drive"),
            StorageType::Dropbox => write!(f, "dropbox"),
            StorageType::IPFS => write!(f, "ipfs"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct FileMetadata {
    pub id: String,
    pub name: String,
    pub size: usize,
    pub modified_time: Option<String>,
}
