use std::sync::Arc;

use anyhow::Result;
use quantization::quantization::Quantizer;
use utils::file_io::env::Env;

use crate::ivf::block_based::index::BlockBasedIvf;

pub struct IvfReader {
    base_directory: String,
    index_offset: usize,
    vector_offset: usize,
}

impl IvfReader {
    pub fn new(base_directory: String) -> Self {
        Self::new_with_offset(base_directory, 0, 0)
    }

    pub fn new_with_offset(
        base_directory: String,
        index_offset: usize,
        vector_offset: usize,
    ) -> Self {
        Self {
            base_directory,
            index_offset,
            vector_offset,
        }
    }

    pub async fn read<Q: Quantizer>(&self, env: Arc<Box<dyn Env>>) -> Result<BlockBasedIvf<Q>>
    where
        Q::QuantizedT: Send + Sync,
    {
        BlockBasedIvf::<Q>::new_with_offset(
            env,
            self.base_directory.clone(),
            self.index_offset,
            self.vector_offset,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use compression::noc::PlainEncoder;
    use quantization::noq::NoQuantizer;
    use quantization::quantization::WritableQuantizer;
    use tempdir::TempDir;
    use utils::distance::l2::L2DistanceCalculator;
    use utils::file_io::env::{DefaultEnv, EnvConfig, FileType};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::ivf::builder::{IvfBuilder, IvfBuilderConfig};
    use crate::ivf::writer::IvfWriter;

    fn create_env() -> Arc<Box<dyn Env>> {
        let config = EnvConfig {
            file_type: FileType::CachedStandard,
            ..EnvConfig::default()
        };
        Arc::new(Box::new(DefaultEnv::new(config)))
    }

    #[tokio::test]
    async fn test_ivf_reader_read() {
        let temp_dir =
            TempDir::new("test_ivf_reader_read").expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();
        let num_clusters = 10;
        let num_vectors = 100;
        let num_features = 4;
        let file_size = 4096;

        let quantizer = NoQuantizer::<L2DistanceCalculator>::new(num_features);
        let quantizer_directory = format!("{}/quantizer", base_directory);
        std::fs::create_dir_all(&quantizer_directory).unwrap();
        quantizer.write_to_directory(&quantizer_directory).unwrap();

        let writer = IvfWriter::<_, PlainEncoder, L2DistanceCalculator>::new(
            base_directory.clone(),
            quantizer,
        );

        let mut builder: IvfBuilder<L2DistanceCalculator> = IvfBuilder::new(IvfBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
            num_clusters,
            num_data_points_for_clustering: num_vectors,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,
            base_directory: base_directory.clone(),
            memory_size: 1024,
            file_size,
            num_features,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        })
        .expect("Failed to create builder");

        for i in 0..num_vectors {
            builder
                .add_vector(i as u128, &generate_random_vector(num_features))
                .unwrap();
        }
        builder.build().unwrap();
        writer.write(&mut builder, false).unwrap();

        let env = create_env();
        let reader = IvfReader::new(base_directory.clone());
        let ivf = reader
            .read::<NoQuantizer<L2DistanceCalculator>>(env)
            .await
            .expect("Failed to read index file");

        assert_eq!(ivf.num_clusters(), num_clusters);
    }
}
