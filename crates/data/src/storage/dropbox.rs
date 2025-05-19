use crate::storage::{FileMetadata, FileUploader, StorageType, UploadOptions};
use anyhow::Result;
use reqwest::Client;
use serde_json::json;

/// Default storage folder name
pub const DEFAULT_FOLDER: &str = "alith";
/// Environment variable name for custom folder configuration
pub const DROPBOX_DEFAULT_FOLDER_ENV: &str = "DROPBOX_DEFAULT_FOLDER";

#[derive(Debug, Clone)]
pub struct DropboxUploader {
    client: Client,
    pub folder: String,
}

impl Default for DropboxUploader {
    fn default() -> Self {
        Self {
            client: Default::default(),
            folder: get_folder::<&str>(None),
        }
    }
}

impl DropboxUploader {
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

    pub async fn get_share_link<S: AsRef<str>>(&self, token: S, file_path: S) -> Result<String> {
        let list_response = self
            .client
            .post("https://api.dropboxapi.com/2/sharing/list_shared_links")
            .bearer_auth(token.as_ref())
            .header("Content-Type", "application/json")
            .json(&json!({
                "path": file_path.as_ref(),
                "direct_only": true
            }))
            .send()
            .await?;

        let list_result: serde_json::Value = list_response.json().await?;
        let existing_link = list_result["links"].as_array().and_then(|links| {
            links
                .iter()
                .find(|l| l["path_lower"] == file_path.as_ref().to_lowercase())
        });

        let share_link = match existing_link {
            Some(link) => link["url"].as_str().unwrap().to_string(),
            None => {
                let create_response = self
                    .client
                    .post("https://api.dropboxapi.com/2/sharing/create_shared_link_with_settings")
                    .bearer_auth(token.as_ref())
                    .json(&json!({
                        "path": file_path.as_ref(),
                        "settings": {
                            "allow_download": true
                        }
                    }))
                    .header("Content-Type", "application/json")
                    .send()
                    .await?;

                let create_result: serde_json::Value = create_response.json().await?;
                create_result["url"]
                    .as_str()
                    .ok_or(anyhow::anyhow!("Missing url in create response"))?
                    .to_string()
            }
        };

        Ok(share_link.replace("dl=0", "dl=1"))
    }
}

#[async_trait::async_trait]
impl FileUploader for DropboxUploader {
    async fn upload(&self, opts: UploadOptions) -> Result<FileMetadata> {
        let UploadOptions { name, data, token } = opts;
        let size = data.len();

        let upload_path = format!("/{}/{}", self.folder, name);

        let api_args = json!({
            "path": upload_path,
            "mode": { ".tag": "add" },
            "autorename": true,
            "mute": false
        });

        let upload_response = self
            .client
            .post("https://content.dropboxapi.com/2/files/upload")
            .bearer_auth(token)
            .header("Dropbox-API-Arg", api_args.to_string())
            .header("Content-Type", "application/octet-stream")
            .body(data)
            .send()
            .await?;

        let upload_result: serde_json::Value = upload_response.json().await?;

        if upload_result.get("error").is_some() {
            return Err(anyhow::anyhow!(
                "Upload failed: {}",
                upload_result.to_string()
            ));
        }

        let path_lower = upload_result["path_lower"]
            .as_str()
            .ok_or(anyhow::anyhow!("Missing path_lower in response"))?;
        let name = upload_result["name"]
            .as_str()
            .ok_or(anyhow::anyhow!("Missing name in response"))?;

        let server_modified = upload_result["server_modified"]
            .as_str()
            .ok_or(anyhow::anyhow!("Missing server_modifield in response"))?;

        Ok(FileMetadata {
            id: path_lower.to_string(),
            name: name.to_string(),
            size,
            modified_time: Some(server_modified.to_string()),
        })
    }

    #[inline]
    fn storage_type(&self) -> StorageType {
        StorageType::Dropbox
    }
}

#[inline]
fn get_folder<S: AsRef<str>>(folder: Option<S>) -> String {
    folder.map(|s| s.as_ref().to_string()).unwrap_or_else(|| {
        std::env::var(DROPBOX_DEFAULT_FOLDER_ENV).unwrap_or_else(|_| DEFAULT_FOLDER.to_string())
    })
}
