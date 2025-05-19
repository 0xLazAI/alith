use crate::storage::{FileMetadata, FileUploader, StorageType, UploadOptions};
use anyhow::{Context, Result};
use reqwest::{Client, multipart};
use serde::{Deserialize, Serialize};

// IPFS Gateway Links

pub const IPFS_LINK: &str = "https://ipfs.io";
pub const IPFS_DWEB_LINK: &str = "https://dweb.link";
pub const IPFS_W3S_LINK: &str = "https://w3s.link";
pub const IPFS_TRUSTLESS_GATEWAY_LINK: &str = "https://trustless-gateway.link";
pub const IPFS_4EVERLAND_LINK: &str = "https://4everland.io";
pub const IPFS_PINATA_CLOUD_LINK: &str = "https://gateway.pinata.cloud";
pub const IPFS_NFT_STORAGE_LINK: &str = "https://nftstorage.link";

// IPFS Secret Env Vars

pub const IPFS_GATEWAY_ENV: &str = "IPFS_GATEWAY";
pub const IPFS_API_KEY_ENV: &str = "IPFS_API_KEY";
pub const IPFS_API_SECRET_ENV: &str = "IPFS_API_SECRET_KEY";
pub const IPFS_JWT_ENV: &str = "IPFS_JWT";

#[derive(Debug, Clone, Default)]
pub struct PinataIPFS {
    client: Client,
}

impl PinataIPFS {
    pub fn new<S: AsRef<str>>() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn get_share_link<S: AsRef<str>>(&self, _token: S, cid: S) -> Result<String> {
        Ok(format!(
            "https://gateway.pinata.cloud/ipfs/{}?download=true",
            cid.as_ref()
        ))
    }
}

#[async_trait::async_trait]
impl FileUploader for PinataIPFS {
    async fn upload(&self, opts: UploadOptions) -> Result<FileMetadata> {
        let UploadOptions { name, data, token } = opts;
        let url = "https://uploads.pinata.cloud/v3/files";
        let file_part = multipart::Part::text(String::from_utf8(data)?)
            .file_name(name)
            .mime_str("text/plain")?;
        let form = multipart::Form::new()
            .part("file", file_part)
            .text("network", "public");
        let response = self
            .client
            .post(url)
            .multipart(form)
            .bearer_auth(token.as_str())
            .send()
            .await
            .context("Failed to send upload request")?;
        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Pinata IPFS API error: {}", error_text);
        }
        let resp = response.json::<PinataUploadResponse>().await?;
        Ok(resp.data.into())
    }

    #[inline]
    fn storage_type(&self) -> StorageType {
        StorageType::IPFS
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PinataUploadResponse {
    pub data: PinataFileDetails,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct PinataFileDetails {
    pub accept_duplicates: bool,
    pub is_duplicate: bool,
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub cid: String,
    pub size: usize,
    pub number_of_files: usize,
    pub mime_type: String,
    pub group_id: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    pub network: String,
    pub streamable: bool,
    pub vectorized: bool,
}

impl From<PinataFileDetails> for FileMetadata {
    fn from(value: PinataFileDetails) -> Self {
        FileMetadata {
            id: value.cid,
            name: value.name,
            size: value.size,
            modified_time: Some(value.updated_at),
        }
    }
}
