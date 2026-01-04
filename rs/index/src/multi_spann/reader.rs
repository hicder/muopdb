use std::sync::Arc;

use anyhow::Result;
use memmap2::Mmap;
use quantization::quantization::Quantizer;
use utils::file_io::env::Env;

use crate::multi_spann::index::MultiSpannIndex;

pub struct MultiSpannReader {
    base_directory: String,
}

impl MultiSpannReader {
    /// Creates a new `MultiSpannReader` for the specified base directory.
    ///
    /// # Arguments
    /// * `base_directory` - The directory where the multi-user SPANN index is stored.
    ///
    /// # Returns
    /// * `Self` - A new `MultiSpannReader` instance.
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    /// Reads and initializes a `MultiSpannIndex` from disk using Env abstraction.
    ///
    /// # Arguments
    /// * `ivf_type` - The encoding type used for IVF posting lists.
    /// * `num_features` - The number of dimensions in the vectors.
    /// * `env` - The environment for file I/O.
    ///
    /// # Returns
    /// * `Result<MultiSpannIndex<Q>>` - The initialized multi-user index or an error.
    pub async fn read<Q: Quantizer>(
        &self,
        num_features: usize,
        env: Arc<Box<dyn Env>>,
    ) -> Result<MultiSpannIndex<Q>> {
        let user_index_info_file_path = format!("{}/user_index_info", self.base_directory);
        let user_index_info_file = std::fs::OpenOptions::new()
            .read(true)
            .open(user_index_info_file_path)?;

        let user_index_info_mmap = unsafe { Mmap::map(&user_index_info_file)? };
        MultiSpannIndex::<Q>::new(
            self.base_directory.clone(),
            user_index_info_mmap,
            num_features,
            env,
        )
        .await
    }
}

#[cfg(test)]
mod tests {

    use config::collection::CollectionConfig;
    use config::enums::QuantizerType;
    use config::search_params::SearchParams;
    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::DefaultEnv;

    use super::*;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::writer::MultiSpannWriter;

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = utils::file_io::env::EnvConfig {
            file_type: utils::file_io::env::FileType::CachedStandard,
            ..utils::file_io::env::EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

    #[tokio::test]
    async fn test_multi_spann_reader() -> Result<()> {
        let temp_dir = tempdir::TempDir::new("test_multi_spann_reader")?;
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = 4;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())?;
        multi_spann_builder.insert(0, 1, &[1.0, 2.0, 3.0, 4.0])?;
        multi_spann_builder.insert(0, 2, &[5.0, 6.0, 7.0, 8.0])?;
        multi_spann_builder.insert(1, 3, &[9.0, 10.0, 11.0, 12.0])?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(4, env)
            .await?;

        let params = SearchParams::new(3, 100, false);

        let result = multi_spann_index
            .search_for_user(0, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        let result = multi_spann_index
            .search_for_user(1, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_spann_reader_pq() -> Result<()> {
        let temp_dir = tempdir::TempDir::new("test_multi_spann_reader_pq")?;
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = 4;
        spann_builder_config.product_quantization_subvector_dimension = 2;
        spann_builder_config.product_quantization_num_bits = 0;
        spann_builder_config.product_quantization_num_training_rows = 1;
        spann_builder_config.product_quantization_batch_size = 0;
        spann_builder_config.quantization_type = QuantizerType::ProductQuantizer;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())?;
        multi_spann_builder.insert(0, 1, &[1.0, 2.0, 3.0, 4.0])?;
        multi_spann_builder.insert(0, 2, &[5.0, 6.0, 7.0, 8.0])?;
        multi_spann_builder.insert(1, 3, &[9.0, 10.0, 11.0, 12.0])?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;

        let env = create_env();
        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>(4, env)
            .await?;

        let params = SearchParams::new(3, 100, false);

        let result = multi_spann_index
            .search_for_user(0, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        let result = multi_spann_index
            .search_for_user(1, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_spann_reader_async() -> Result<()> {
        let temp_dir = tempdir::TempDir::new("test_multi_spann_reader_async")?;
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = 4;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())?;
        multi_spann_builder.insert(0, 1, &[1.0, 2.0, 3.0, 4.0])?;
        multi_spann_builder.insert(0, 2, &[5.0, 6.0, 7.0, 8.0])?;
        multi_spann_builder.insert(1, 3, &[9.0, 10.0, 11.0, 12.0])?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;

        let mut env_config = utils::file_io::env::EnvConfig::default();
        env_config.file_type = utils::file_io::env::FileType::MMap;
        let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(env_config)));

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(4, env)
            .await?;

        let params = SearchParams::new(3, 100, false);

        let result = multi_spann_index
            .search_for_user(0, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        let result = multi_spann_index
            .search_for_user(1, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_multi_spann_reader_async_pq() -> Result<()> {
        let temp_dir = tempdir::TempDir::new("test_multi_spann_reader_async_pq")?;
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let mut spann_builder_config = CollectionConfig::default_test_config();
        spann_builder_config.num_features = 4;
        spann_builder_config.product_quantization_subvector_dimension = 2;
        spann_builder_config.product_quantization_num_bits = 0;
        spann_builder_config.product_quantization_num_training_rows = 1;
        spann_builder_config.product_quantization_batch_size = 0;
        spann_builder_config.quantization_type = QuantizerType::ProductQuantizer;
        let mut multi_spann_builder =
            MultiSpannBuilder::new(spann_builder_config, base_directory.clone())?;
        multi_spann_builder.insert(0, 1, &[1.0, 2.0, 3.0, 4.0])?;
        multi_spann_builder.insert(0, 2, &[5.0, 6.0, 7.0, 8.0])?;
        multi_spann_builder.insert(1, 3, &[9.0, 10.0, 11.0, 12.0])?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;

        let mut env_config = utils::file_io::env::EnvConfig::default();
        env_config.file_type = utils::file_io::env::FileType::MMap;
        let env: Arc<Box<dyn Env>> = Arc::new(Box::new(DefaultEnv::new(env_config)));

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>(4, env)
            .await
            .unwrap();

        let params = SearchParams::new(3, 100, false);

        let result = multi_spann_index
            .search_for_user(0, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 1);

        let result = multi_spann_index
            .search_for_user(1, vec![1.0, 2.0, 3.0, 4.0], &params, None)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].doc_id, 3);

        Ok(())
    }
}
