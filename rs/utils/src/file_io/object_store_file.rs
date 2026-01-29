use std::fs::Metadata;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use tracing::debug;
use url::Url;

use crate::file_io::FileIO;

/// A [`FileIO`] implementation that uses an [`ObjectStore`] for reading.
pub struct ObjectStoreFileIO {
    store: Arc<dyn ObjectStore>,
    path: Path,
    file_length: u64,
}

impl ObjectStoreFileIO {
    pub async fn new(
        url_str: &str,
        endpoint: Option<&str>,
        region: Option<&str>,
        access_key: Option<&str>,
        secret_key: Option<&str>,
    ) -> Result<Self> {
        let url =
            Url::parse(url_str).map_err(|e| anyhow!("Failed to parse URL {}: {}", url_str, e))?;
        if url.scheme() != "s3" {
            return Err(anyhow!("Unsupported scheme: {}", url.scheme()));
        }

        let bucket = url
            .host_str()
            .ok_or_else(|| anyhow!("Bucket not found in URL"))?;
        let path_str = url.path().trim_start_matches('/');
        let path = Path::from(path_str);

        let mut builder = AmazonS3Builder::from_env().with_bucket_name(bucket);

        if let Some(endpoint) = endpoint {
            builder = builder.with_endpoint(endpoint);
            builder = builder.with_allow_http(true); // Useful for local testing with MinIO
        }
        if let Some(region) = region {
            builder = builder.with_region(region);
        }
        if let Some(access_key) = access_key {
            builder = builder.with_access_key_id(access_key);
        }
        if let Some(secret_key) = secret_key {
            builder = builder.with_secret_access_key(secret_key);
        }

        let store = Arc::new(builder.build()?);

        // Get initial metadata to know the file length
        let meta = store
            .head(&path)
            .await
            .map_err(|e| anyhow!("Failed to get head for {}: {}", url_str, e))?;

        Ok(Self {
            store,
            path,
            file_length: meta.size as u64,
        })
    }
}

#[async_trait]
impl FileIO for ObjectStoreFileIO {
    async fn read(&self, offset: u64, length: u64) -> Result<Vec<u8>> {
        debug!(
            "[OBJECT_STORE] Reading: {} offset {} length {}",
            self.path, offset, length
        );
        let range = offset..(offset + length);
        let bytes = self
            .store
            .get_range(&self.path, range)
            .await
            .map_err(|e| anyhow!("Failed to read from object store: {}", e))?;
        Ok(bytes.to_vec())
    }

    async fn metadata(&self) -> Result<Metadata> {
        // This is tricky because object_store::ObjectMeta doesn't easily convert to std::fs::Metadata.
        // For MuopDB's current needs, we might only need size.
        // If we really need Metadata, we might need to mock it or update the trait.
        // For now, we'll return an error if it's called or just ignore it if possible.
        Err(anyhow!("Metadata not supported for ObjectStoreFileIO"))
    }

    async fn file_length(&self) -> Result<u64> {
        Ok(self.file_length)
    }

    fn get_block_size(&self) -> usize {
        4096 // Default
    }
}
