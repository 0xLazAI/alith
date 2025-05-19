use crate::storage::{FileMetadata, FileUploader, StorageType, UploadOptions};
use anyhow::{Context, Result};
use reqwest::{Client, multipart};
use serde::{Deserialize, Serialize};

/// Default storage folder name
pub const DEFAULT_FOLDER: &str = "alith";
/// Environment variable name for custom folder configuration
pub const GOOGLE_DRIVE_DEFAULT_FOLDER_ENV: &str = "GOOGLE_DRIVE_DEFAULT_FOLDER";
/// Google drive base url.
pub const GOOGLE_DRIVE_URL: &str = "https://www.googleapis.com/drive/v3/files";

#[derive(Debug, Clone, Default)]
pub struct GoogleDriveUploader {
    client: Client,
    pub folder: String,
}

impl GoogleDriveUploader {
    pub fn new<S: AsRef<str>>(folder: Option<S>) -> Self {
        Self {
            client: Client::new(),
            folder: get_folder(folder),
        }
    }

    pub fn with_folder<S: AsRef<str>>(mut self, folder: S) -> Self {
        self.folder = folder.as_ref().to_string();
        self
    }

    pub async fn find_or_create_folder<S: AsRef<str>>(
        &self,
        token: S,
        folder: S,
    ) -> Result<String> {
        let query = [
            (
                "q",
                format!(
                    "name = '{}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
                    folder.as_ref()
                ),
            ),
            ("fields", "files(id, name)".to_string()),
            ("pageSize", "1".to_string()),
        ];

        let response = self
            .client
            .get(GOOGLE_DRIVE_URL)
            .bearer_auth(token.as_ref())
            .query(&query)
            .send()
            .await?;

        let data: serde_json::Value = response.json().await?;
        if let Some(files) = data.get("files").and_then(|f| f.as_array()) {
            if !files.is_empty() {
                return files[0]["id"]
                    .as_str()
                    .map(|s| s.to_string())
                    .ok_or(anyhow::anyhow!("Invalid folder ID format"));
            }
        }

        let metadata = serde_json::json!({
            "name": folder.as_ref(),
            "mimeType": "application/vnd.google-apps.folder"
        });

        let response = self
            .client
            .post(GOOGLE_DRIVE_URL)
            .bearer_auth(token.as_ref())
            .json(&metadata)
            .send()
            .await
            .context("Failed to create folder")?;

        let folder_data: serde_json::Value = response.json().await?;
        folder_data["id"]
            .as_str()
            .map(|s| s.to_string())
            .context("Invalid folder ID in response")
    }

    pub async fn fetch_file_details<S: AsRef<str>>(
        &self,
        token: S,
        file_id: S,
    ) -> Result<FileDetails> {
        let url = format!(
            "{}/{}?fields=id,name,md5Checksum,modifiedTime",
            GOOGLE_DRIVE_URL,
            file_id.as_ref()
        );

        let response = self
            .client
            .get(&url)
            .bearer_auth(token.as_ref())
            .send()
            .await
            .context("Failed to fetch file details")?;

        response
            .json()
            .await
            .context("Failed to parse file details")
    }

    pub async fn get_share_link<S: AsRef<str>>(&self, token: S, file_id: S) -> Result<String> {
        let token = token.as_ref();
        let file_id = file_id.as_ref().to_string();
        self.update_file_permissions(token, &file_id).await?;
        Ok(format!(
            "https://drive.google.com/uc?export=download&id={}",
            file_id
        ))
    }

    pub async fn update_file_permissions<S: AsRef<str>>(&self, token: S, file_id: S) -> Result<()> {
        let url = format!("{}/{}/permissions", GOOGLE_DRIVE_URL, file_id.as_ref());

        let body = serde_json::json!({
            "role": "reader",
            "type": "anyone"
        });

        let response = self
            .client
            .post(&url)
            .bearer_auth(token.as_ref())
            .json(&body)
            .send()
            .await
            .context("Failed to update permissions")?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            anyhow::bail!("Permission update failed: {}", error_text);
        }

        Ok(())
    }
}

#[async_trait::async_trait]
impl FileUploader for GoogleDriveUploader {
    async fn upload(&self, opts: UploadOptions) -> Result<FileMetadata> {
        let UploadOptions { name, data, token } = opts;
        let size = data.len();
        let folder_id = self
            .find_or_create_folder(token.as_str(), self.folder.as_str())
            .await?;
        let url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart";

        let metadata = DriveFileMetadata {
            name: name.clone(),
            parents: vec![folder_id],
        };
        let metadata_part = multipart::Part::text(serde_json::to_string(&metadata)?)
            .mime_str("application/json")?;

        let file_part = multipart::Part::bytes(data).file_name(name);

        let form = multipart::Form::new()
            .part("metadata", metadata_part)
            .part("file", file_part);

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
            anyhow::bail!("Google Drive API error: {}", error_text);
        }

        let file_data: serde_json::Value = response.json().await?;
        let file_id = file_data["id"]
            .as_str()
            .context("Invalid response format: missing file ID")?;

        let mut file_metadata: FileMetadata = self
            .fetch_file_details(token.as_str(), file_id)
            .await?
            .into();
        file_metadata.size = size;
        Ok(file_metadata)
    }

    #[inline]
    fn storage_type(&self) -> StorageType {
        StorageType::GoogleDrive
    }
}

#[derive(Serialize)]
struct DriveFileMetadata {
    name: String,
    parents: Vec<String>,
}

#[derive(Deserialize, Debug)]
pub struct FileDetails {
    pub id: String,
    pub name: String,
    #[serde(rename = "md5Checksum")]
    pub md5_checksum: Option<String>,
    #[serde(rename = "modifiedTime")]
    pub modified_time: String,
}

impl From<FileDetails> for FileMetadata {
    fn from(value: FileDetails) -> Self {
        FileMetadata {
            id: value.id,
            name: value.name,
            size: 0,
            modified_time: Some(value.modified_time),
        }
    }
}

#[inline]
fn get_folder<S: AsRef<str>>(folder: Option<S>) -> String {
    folder.map(|s| s.as_ref().to_string()).unwrap_or_else(|| {
        std::env::var(GOOGLE_DRIVE_DEFAULT_FOLDER_ENV)
            .unwrap_or_else(|_| DEFAULT_FOLDER.to_string())
    })
}
