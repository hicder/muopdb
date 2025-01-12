use anyhow::Result;
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

    pub fn read<Q: Quantizer>(&self) -> Result<MultiSpannIndex<Q>> {
        let user_index_info_file_path = format!("{}/user_index_info", self.base_directory);
        let user_index_info_file = std::fs::OpenOptions::new()
            .read(true)
            .open(user_index_info_file_path)?;

        let user_index_info_mmap = unsafe { Mmap::map(&user_index_info_file)? };
        MultiSpannIndex::<Q>::new(self.base_directory.clone(), user_index_info_mmap)
    }
}

#[cfg(test)]
mod tests {

    use quantization::noq::noq::NoQuantizer;
    use utils::distance::l2::L2DistanceCalculator;

    use super::*;
    use crate::index::Searchable;
    use crate::multi_spann::builder::MultiSpannBuilder;
    use crate::multi_spann::writer::MultiSpannWriter;
    use crate::spann::builder::SpannBuilderConfig;
    use crate::utils::SearchContext;

    #[test]
    fn test_multi_spann_reader() -> Result<()> {
        let temp_dir = tempdir::TempDir::new("test_multi_spann_reader")?;
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let mut spann_builder_config = SpannBuilderConfig::default();
        spann_builder_config.num_features = 4;
        spann_builder_config.base_directory = base_directory.clone();
        let mut multi_spann_builder = MultiSpannBuilder::new(spann_builder_config)?;
        multi_spann_builder.insert(0, 1, &[1.0, 2.0, 3.0, 4.0])?;
        multi_spann_builder.insert(0, 2, &[5.0, 6.0, 7.0, 8.0])?;
        multi_spann_builder.insert(1, 3, &[9.0, 10.0, 11.0, 12.0])?;
        multi_spann_builder.build()?;

        let multi_spann_writer = MultiSpannWriter::new(base_directory.clone());
        multi_spann_writer.write(&mut multi_spann_builder)?;

        let multi_spann_reader = MultiSpannReader::new(base_directory);
        // TODO(tyb): use config instead of hardcoding
        let multi_spann_index = multi_spann_reader.read::<NoQuantizer<L2DistanceCalculator>>()?;

        let result = multi_spann_index
            .search_with_id(
                0,
                &[1.0, 2.0, 3.0, 4.0],
                3,
                100,
                &mut SearchContext::new(false),
            )
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 1);

        let result = multi_spann_index
            .search_with_id(
                1,
                &[1.0, 2.0, 3.0, 4.0],
                3,
                100,
                &mut SearchContext::new(false),
            )
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, 3);

        Ok(())
    }
}
