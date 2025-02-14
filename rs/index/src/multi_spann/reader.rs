use anyhow::Result;
use config::enums::IntSeqEncodingType;
use memmap2::Mmap;
use quantization::quantization::Quantizer;

use crate::multi_spann::index::MultiSpannIndex;

pub struct MultiSpannReader {
    base_directory: String,
}

impl MultiSpannReader {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn read<Q: Quantizer>(&self, ivf_type: IntSeqEncodingType) -> Result<MultiSpannIndex<Q>> {
        let user_index_info_file_path = format!("{}/user_index_info", self.base_directory);
        let user_index_info_file = std::fs::OpenOptions::new()
            .read(true)
            .open(user_index_info_file_path)?;

        let user_index_info_mmap = unsafe { Mmap::map(&user_index_info_file)? };
        MultiSpannIndex::<Q>::new(self.base_directory.clone(), user_index_info_mmap, ivf_type)
    }
}

#[cfg(test)]
mod tests {

    use config::collection::CollectionConfig;
    use config::enums::QuantizerType;
    use quantization::noq::noq::NoQuantizer;
    use quantization::pq::pq::ProductQuantizer;
    use utils::distance::l2::L2DistanceCalculator;

    use super::*;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::writer::MultiSpannWriter;

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

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<NoQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)?;

        let result = multi_spann_index
            .search_with_id(0, vec![1.0, 2.0, 3.0, 4.0], 3, 100, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].id, 1);

        let result = multi_spann_index
            .search_with_id(1, vec![1.0, 2.0, 3.0, 4.0], 3, 100, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].id, 3);

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

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        let multi_spann_index = multi_spann_reader
            .read::<ProductQuantizer<L2DistanceCalculator>>(IntSeqEncodingType::PlainEncoding)?;

        let result = multi_spann_index
            .search_with_id(0, vec![1.0, 2.0, 3.0, 4.0], 3, 100, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].id, 1);

        let result = multi_spann_index
            .search_with_id(1, vec![1.0, 2.0, 3.0, 4.0], 3, 100, false)
            .await
            .unwrap();
        assert_eq!(result.id_with_scores.len(), 1);
        assert_eq!(result.id_with_scores[0].id, 3);

        Ok(())
    }
}
