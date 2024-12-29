use anyhow::{Ok, Result};
use index::hnsw::builder::HnswBuilder;
use index::hnsw::writer::HnswWriter;
use index::ivf::builder::{IvfBuilder, IvfBuilderConfig};
use index::ivf::writer::IvfWriter;
use index::spann::builder::{SpannBuilder, SpannBuilderConfig};
use index::spann::writer::SpannWriter;
use log::{debug, info};
use quantization::noq::noq::{NoQuantizer, NoQuantizerConfig, NoQuantizerWriter};
use quantization::noq::noq_builder::NoQuantizerBuilder;
use quantization::pq::pq::{ProductQuantizer, ProductQuantizerConfig, ProductQuantizerWriter};
use quantization::pq::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
use quantization::quantization::Quantizer;
use rand::seq::SliceRandom;
use utils::distance::dot_product::DotProductDistanceCalculator;
use utils::distance::l2::L2DistanceCalculator;
use utils::{CalculateSquared, DistanceCalculator};

use crate::config::{
    DistanceType, HnswConfigWithBase, IndexWriterConfig, IvfConfigWithBase, QuantizerType, SpannConfigWithBase
};
use crate::input::Input;

pub struct IndexWriter {
    config: IndexWriterConfig,
    output_root: String,
}

impl IndexWriter {
    pub fn new(config: IndexWriterConfig) -> Result<Self> {
        let base_config = match config.clone() {
            IndexWriterConfig::Hnsw(hnsw_config) => hnsw_config.base_config,
            IndexWriterConfig::Ivf(ivf_config) => ivf_config.base_config,
            IndexWriterConfig::Spann(hnsw_ivf_config) => hnsw_ivf_config.base_config,
        };

        let index_type_str = format!("{:?}", base_config.index_type).to_lowercase();
        let output_root = format!("{}/{}", base_config.output_path, index_type_str);
        std::fs::create_dir_all(&output_root)?;

        Ok(Self {
            config,
            output_root,
        })
    }

    fn get_sorted_random_rows(num_rows: usize, num_random_rows: usize) -> Vec<u64> {
        let mut v = (0..num_rows).map(|x| x as u64).collect::<Vec<_>>();
        v.shuffle(&mut rand::thread_rng());
        let mut ret = v.into_iter().take(num_random_rows).collect::<Vec<u64>>();
        ret.sort();
        ret
    }

    fn write_quantizer_and_build_hnsw_index<T: Quantizer, F: Fn(&String, &T) -> Result<()>>(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &HnswConfigWithBase,
        quantizer: T,
        writer_fn: F,
    ) -> Result<()> {
        info!("Start writing product quantizer");
        let path = &self.output_root;

        let quantizer_directory = format!("{}/quantizer", path);
        std::fs::create_dir_all(&quantizer_directory)?;

        // Use the provided writer function to write the quantizer
        writer_fn(&quantizer_directory, &quantizer)?;

        info!("Start building index");
        let vector_directory = format!("{}/vectors", path);
        std::fs::create_dir_all(&vector_directory)?;

        let mut hnsw_builder = HnswBuilder::<T>::new(
            index_builder_config.hnsw_config.max_num_neighbors,
            index_builder_config.hnsw_config.num_layers,
            index_builder_config.hnsw_config.ef_construction,
            index_builder_config.base_config.max_memory_size,
            index_builder_config.base_config.file_size,
            index_builder_config.base_config.dimension
                / index_builder_config.quantizer_config.subvector_dimension,
            quantizer,
            vector_directory.clone(),
        );

        input.reset();
        while input.has_next() {
            let row = input.next();
            hnsw_builder.insert(row.id, row.data)?;
            if row.id % 10000 == 0 {
                debug!("Inserted {} rows", row.id);
            }
        }

        std::fs::create_dir_all(&path)?;

        info!("Start writing index");
        let hnsw_writer = HnswWriter::new(path.to_string());
        hnsw_writer.write(&mut hnsw_builder, index_builder_config.base_config.reindex)?;

        // Cleanup tmp directory. It's ok to fail
        std::fs::remove_dir_all(&vector_directory).unwrap_or_default();

        Ok(())
    }

    fn build_hnsw_pq(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &HnswConfigWithBase,
    ) -> Result<()> {
        // Create and train product quantizer
        let pq_config = ProductQuantizerConfig {
            dimension: index_builder_config.base_config.dimension,
            subvector_dimension: index_builder_config.quantizer_config.subvector_dimension,
            num_bits: index_builder_config.quantizer_config.num_bits,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: index_builder_config.quantizer_config.max_iteration,
            batch_size: index_builder_config.quantizer_config.batch_size,
        };

        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        info!("Start training product quantizer");
        let sorted_random_rows = Self::get_sorted_random_rows(
            input.num_rows(),
            index_builder_config.quantizer_config.num_training_rows,
        );

        for row_idx in sorted_random_rows {
            input.skip_to(row_idx as usize);
            pq_builder.add(input.next().data.to_vec());
        }

        let pq = pq_builder.build(format!("{}/pq_tmp", &self.output_root))?;

        // Define the writer function for ProductQuantizer
        let pq_writer_fn = |directory: &String, pq: &ProductQuantizer| {
            let pq_writer = ProductQuantizerWriter::new(directory.clone());
            pq_writer.write(pq)
        };

        self.write_quantizer_and_build_hnsw_index(input, index_builder_config, pq, pq_writer_fn)
    }

    fn build_hnsw_noq(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &HnswConfigWithBase,
    ) -> Result<()> {
        // Create NoQuantizer
        let noq_config = NoQuantizerConfig {
            dimension: index_builder_config.base_config.dimension,
        };

        let mut noq_builder = NoQuantizerBuilder::new(noq_config);

        let noq = noq_builder.build()?;

        // Define the writer function for NoQuantizer
        let noq_writer_fn = |directory: &String, noq: &NoQuantizer| {
            let noq_writer = NoQuantizerWriter::new(directory.clone());
            noq_writer.write(noq)
        };

        self.write_quantizer_and_build_hnsw_index(input, index_builder_config, noq, noq_writer_fn)
    }

    fn do_build_hnsw_index(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &HnswConfigWithBase,
    ) -> Result<()> {
        match index_builder_config.quantizer_config.quantizer_type {
            QuantizerType::ProductQuantizer => {
                self.build_hnsw_pq(input, index_builder_config)?;
            }
            QuantizerType::NoQuantizer => {
                self.build_hnsw_noq(input, index_builder_config)?;
            }
        };
        Ok(())
    }

    fn write_quantizer_and_build_ivf_index<T, D, F>(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &IvfConfigWithBase,
        quantizer: T,
        writer_fn: F,
    ) -> Result<()> 
    where 
        T: Quantizer,
        D: DistanceCalculator + CalculateSquared + Send + Sync,
        F: Fn(&String, &T)  -> Result<()>,
    {
        info!("Start writing product quantizer");
        let path = &self.output_root;

        let quantizer_directory = format!("{}/quantizer", path);
        std::fs::create_dir_all(&quantizer_directory)?;

        // Use the provided writer function to write the quantizer
        writer_fn(&quantizer_directory, &quantizer)?;

        let mut ivf_builder = IvfBuilder::<D>::new(IvfBuilderConfig {
            max_iteration: index_builder_config.ivf_config.max_iteration,
            batch_size: index_builder_config.ivf_config.batch_size,
            num_clusters: index_builder_config.ivf_config.num_clusters,
            num_data_points: index_builder_config.ivf_config.num_data_points,
            max_clusters_per_vector: index_builder_config.ivf_config.max_clusters_per_vector,
            distance_threshold: index_builder_config.ivf_config.distance_threshold,
            base_directory: path.to_string(),
            memory_size: index_builder_config.base_config.max_memory_size,
            file_size: index_builder_config.base_config.file_size,
            num_features: index_builder_config.base_config.dimension,
            tolerance: index_builder_config.ivf_config.tolerance,
            max_posting_list_size: index_builder_config.ivf_config.max_posting_list_size,
        })?;

        input.reset();
        while input.has_next() {
            let row = input.next();
            ivf_builder.add_vector(row.id, row.data)?;
            if row.id % 10000 == 0 {
                debug!("Inserted {} rows", row.id);
            }
        }

        info!("Start building index");
        ivf_builder.build()?;

        std::fs::create_dir_all(&path)?;

        info!("Start writing index");
        //let quantizer = NoQuantizer::new(index_builder_config.base_config.dimension);
        let ivf_writer = IvfWriter::new(path.to_string(), quantizer);
        ivf_writer.write(&mut ivf_builder, index_builder_config.base_config.reindex)?;

        // Cleanup tmp directory. It's ok to fail
        ivf_builder.cleanup()?;

        Ok(())
    }

    fn build_ivf_pq(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &IvfConfigWithBase,
    ) -> Result<()> {
        // Create and train product quantizer
        let pq_config = ProductQuantizerConfig {
            dimension: index_builder_config.base_config.dimension,
            subvector_dimension: index_builder_config.quantizer_config.subvector_dimension,
            num_bits: index_builder_config.quantizer_config.num_bits,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: index_builder_config.quantizer_config.max_iteration,
            batch_size: index_builder_config.quantizer_config.batch_size,
        };

        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        info!("Start training product quantizer");
        let sorted_random_rows = Self::get_sorted_random_rows(
            input.num_rows(),
            index_builder_config.quantizer_config.num_training_rows,
        );

        for row_idx in sorted_random_rows {
            input.skip_to(row_idx as usize);
            pq_builder.add(input.next().data.to_vec());
        }

        let pq = pq_builder.build(format!("{}/pq_tmp", &self.output_root))?;

        // Define the writer function for ProductQuantizer
        let pq_writer_fn = |directory: &String, pq: &ProductQuantizer| {
            let pq_writer = ProductQuantizerWriter::new(directory.clone());
            pq_writer.write(pq)
        };

        match index_builder_config.base_config.index_distance_type {
            DistanceType::DotProduct => self.write_quantizer_and_build_ivf_index::<_, DotProductDistanceCalculator, _>(input, index_builder_config, pq, pq_writer_fn),
            DistanceType::L2 => self.write_quantizer_and_build_ivf_index::<_, L2DistanceCalculator, _>(input, index_builder_config, pq, pq_writer_fn),
        }
        
    }

    fn build_ivf_noq(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &IvfConfigWithBase,
    ) -> Result<()> {
        // Create NoQuantizer
        let noq_config = NoQuantizerConfig {
            dimension: index_builder_config.base_config.dimension,
        };

        let mut noq_builder = NoQuantizerBuilder::new(noq_config);

        let noq = noq_builder.build()?;

        // Define the writer function for NoQuantizer
        let noq_writer_fn = |directory: &String, noq: &NoQuantizer| {
            let noq_writer = NoQuantizerWriter::new(directory.clone());
            noq_writer.write(noq)
        };

        match index_builder_config.base_config.index_distance_type {
            DistanceType::DotProduct => self.write_quantizer_and_build_ivf_index::<_, DotProductDistanceCalculator, _>(input, index_builder_config, noq, noq_writer_fn),
            DistanceType::L2 => self.write_quantizer_and_build_ivf_index::<_, L2DistanceCalculator, _>(input, index_builder_config, noq, noq_writer_fn)
        }
    }

    fn do_build_ivf_index(
        &mut self,
        input: &mut impl Input,
        index_builder_config: &IvfConfigWithBase,
    ) -> Result<()> {
        // Directory structure:
        // ivf_config.base_config.output_path
        // └── ivf
        //     ├── index
        //     ├── quantizer
        //     │   ├── codebook
        //     │   └── product_quantizer_config.yaml
        //     └── vectors
        match index_builder_config.quantizer_config.quantizer_type {
            QuantizerType::ProductQuantizer => {
                self.build_ivf_pq(input, index_builder_config)?;
            }
            QuantizerType::NoQuantizer => {
                self.build_ivf_noq(input, index_builder_config)?;
            }
        };

        Ok(())
    }

    #[allow(unused_variables)]
    fn do_build_ivf_hnsw_index(
        &mut self,
        input: &mut impl Input,
        index_writer_config: &SpannConfigWithBase,
    ) -> Result<()> {
        // Directory structure:
        // hnsw_ivf_config.base_config.output_path
        // ├── centroids
        // │   ├── vector_storage
        // │   └── index
        // ├── ivf
        // │   ├── ivf
        // │   └── centroids
        // └── centroid_quantizer
        //     └── no_quantizer_config.yaml

        let root_path = &self.output_root;

        let spann_config = SpannBuilderConfig {
            max_neighbors: index_writer_config.hnsw_config.max_num_neighbors,
            max_layers: index_writer_config.hnsw_config.num_layers,
            ef_construction: index_writer_config.hnsw_config.ef_construction,
            vector_storage_memory_size: index_writer_config.base_config.max_memory_size,
            vector_storage_file_size: index_writer_config.base_config.file_size,
            num_features: index_writer_config.base_config.dimension,
            max_iteration: index_writer_config.ivf_config.max_iteration,
            batch_size: index_writer_config.ivf_config.batch_size,
            num_clusters: index_writer_config.ivf_config.num_clusters,
            num_data_points: index_writer_config.ivf_config.num_data_points,
            max_clusters_per_vector: index_writer_config.ivf_config.max_clusters_per_vector,
            distance_threshold: index_writer_config.ivf_config.distance_threshold,
            base_directory: root_path.to_string(),
            memory_size: index_writer_config.base_config.max_memory_size,
            file_size: index_writer_config.base_config.file_size,
            tolerance: index_writer_config.ivf_config.tolerance,
            max_posting_list_size: index_writer_config.ivf_config.max_posting_list_size,
            reindex: index_writer_config.base_config.reindex,
        };
        let mut spann_builder = SpannBuilder::new(spann_config)?;

        input.reset();
        while input.has_next() {
            let row = input.next();
            spann_builder.add(row.id, row.data)?;
            if row.id % 10000 == 0 {
                debug!("Inserted {} rows", row.id);
            }
        }

        info!("Start building IVF index");
        spann_builder.build()?;

        let spann_writer = SpannWriter::new(root_path.to_string());
        spann_writer.write(&mut spann_builder)?;

        Ok(())
    }

    // TODO(hicder): Support multiple inputs
    pub fn process(&mut self, input: &mut impl Input) -> Result<()> {
        let cfg = self.config.clone();
        let (base_config, quantizer_config) = match cfg {
            IndexWriterConfig::Hnsw(hnsw_config) => {
                self.do_build_hnsw_index(input, &hnsw_config)?;
                (hnsw_config.base_config, hnsw_config.quantizer_config)
            }
            IndexWriterConfig::Ivf(ivf_config) => {
                self.do_build_ivf_index(input, &ivf_config)?;
                (ivf_config.base_config, ivf_config.quantizer_config)
            }
            IndexWriterConfig::Spann(hnsw_ivf_config) => {
                self.do_build_ivf_hnsw_index(input, &hnsw_ivf_config)?;
                (
                    hnsw_ivf_config.base_config,
                    hnsw_ivf_config.quantizer_config,
                )
            }
        };

        // Finally, write the base config and the quantizer config
        std::fs::write(
            format!("{}/base_config.yaml", self.output_root),
            serde_yaml::to_string(&base_config)?,
        )?;

        std::fs::write(
            format!("{}/quantizer_config.yaml", self.output_root),
            serde_yaml::to_string(&quantizer_config)?,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use rand::Rng;
    use tempdir::TempDir;

    use super::*;
    use crate::config::{BaseConfig, DistanceType, HnswConfig, IndexType, IvfConfig, QuantizerConfig};
    use crate::input::Row;
    // Mock Input implementation for testing
    struct MockInput {
        data: Vec<Vec<f32>>,
        current_index: usize,
    }

    impl MockInput {
        fn new(data: Vec<Vec<f32>>) -> Self {
            Self {
                data,
                current_index: 0,
            }
        }
    }

    impl Input for MockInput {
        fn num_rows(&self) -> usize {
            self.data.len()
        }

        fn skip_to(&mut self, index: usize) {
            self.current_index = index;
        }

        fn next(&mut self) -> Row {
            let row = Row {
                id: self.current_index as u64,
                data: &self.data[self.current_index],
            };
            self.current_index += 1;
            row
        }

        fn has_next(&self) -> bool {
            self.current_index < self.data.len()
        }

        fn reset(&mut self) {
            self.current_index = 0;
        }
    }

    #[test]
    fn test_get_sorted_random_rows() {
        let num_rows = 100;
        let num_random_rows = 50;
        let result = IndexWriter::get_sorted_random_rows(num_rows, num_random_rows);
        assert_eq!(result.len(), num_random_rows);
        for i in 1..result.len() {
            assert!(result[i - 1] <= result[i]);
        }
    }

    #[test]
    fn test_index_writer_process_hnsw() {
        // Setup test data
        let mut rng = rand::thread_rng();
        let dimension = 10;
        let num_rows = 100;
        let data: Vec<Vec<f32>> = (0..num_rows)
            .map(|_| (0..dimension).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let mut mock_input = MockInput::new(data);

        // Create a temporary directory for output
        let temp_dir = TempDir::new("test_index_writer_process_ivf")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Configure IndexWriter
        let base_config = BaseConfig {
            output_path: base_directory.clone(),
            dimension,
            reindex: false,
            max_memory_size: 1024 * 1024 * 1024, // 1 GB
            file_size: 1024 * 1024 * 1024,       // 1 GB
            index_type: IndexType::Hnsw,
            index_distance_type: DistanceType::L2,
        };
        let quantizer_config = QuantizerConfig {
            quantizer_type: QuantizerType::ProductQuantizer,
            quantizer_distance_type: DistanceType::L2,
            subvector_dimension: 2,
            num_bits: 2,
            num_training_rows: 50,

            max_iteration: 10,
            batch_size: 10,
        };
        let hnsw_config = HnswConfig {
            num_layers: 2,
            max_num_neighbors: 10,
            ef_construction: 100,
        };
        let config = IndexWriterConfig::Hnsw(HnswConfigWithBase {
            base_config,
            quantizer_config,
            hnsw_config,
        });

        let mut index_writer = IndexWriter::new(config).expect("Failed to create index writer");

        // Process the input
        index_writer.process(&mut mock_input).unwrap();

        // Check if output directories and files exist
        let hnsw_directory_path = format!("{}/hnsw", base_directory);
        let hnsw_directory = Path::new(&hnsw_directory_path);
        let pq_directory_path = format!("{}/quantizer", hnsw_directory_path);
        let pq_directory = Path::new(&pq_directory_path);
        let hnsw_vector_storage_path =
            format!("{}/vector_storage", hnsw_directory.to_str().unwrap());
        let hnsw_vector_storage = Path::new(&hnsw_vector_storage_path);
        let hnsw_index_path = format!("{}/index", hnsw_directory.to_str().unwrap());
        let hnsw_index = Path::new(&hnsw_index_path);
        assert!(pq_directory.exists());
        assert!(hnsw_directory.exists());
        assert!(hnsw_vector_storage.exists());
        assert!(hnsw_index.exists());
    }

    #[test]
    fn test_index_writer_process_ivf() {
        // Setup test data
        let mut rng = rand::thread_rng();
        let dimension = 10;
        let num_rows = 100;
        let data: Vec<Vec<f32>> = (0..num_rows)
            .map(|_| (0..dimension).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let mut mock_input = MockInput::new(data);

        // Create a temporary directory for output
        let temp_dir = TempDir::new("test_index_writer_process_ivf")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Configure IndexWriter
        let base_config = BaseConfig {
            output_path: base_directory.clone(),
            dimension,
            reindex: false,
            max_memory_size: 1024 * 1024 * 1024, // 1 GB
            file_size: 1024 * 1024 * 1024,       // 1 GB
            index_type: IndexType::Ivf,
            index_distance_type: DistanceType::DotProduct
        };
        let quantizer_config = QuantizerConfig {
            quantizer_type: QuantizerType::ProductQuantizer,
            quantizer_distance_type: DistanceType::L2,
            subvector_dimension: 2,
            num_bits: 2,
            num_training_rows: 50,
            
            max_iteration: 10,
            batch_size: 10,
        };
        let ivf_config = IvfConfig {
            num_clusters: 2,
            num_data_points: 100,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,

            max_iteration: 10,
            batch_size: 10,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        };
        let config = IndexWriterConfig::Ivf(IvfConfigWithBase {
            base_config,
            quantizer_config,
            ivf_config,
        });

        let mut index_writer = IndexWriter::new(config).expect("Failed to create index writer");

        // Process the input
        index_writer.process(&mut mock_input).unwrap();

        // Check if output directories and files exist
        let ivf_directory_path = format!("{}/ivf", base_directory);
        let ivf_directory = Path::new(&ivf_directory_path);
        let ivf_vector_storage_path = format!("{}/vectors", ivf_directory.to_str().unwrap());
        let ivf_vector_storage = Path::new(&ivf_vector_storage_path);
        let ivf_index_path = format!("{}/index", ivf_directory.to_str().unwrap());
        let ivf_index = Path::new(&ivf_index_path);
        assert!(ivf_directory.exists());
        assert!(ivf_vector_storage.exists());
        assert!(ivf_index.exists());
    }

    #[test]
    fn test_index_writer_process_ivf_hnsw() {
        // Setup test data
        let mut rng = rand::thread_rng();
        let dimension = 10;
        let num_rows = 100;
        let data: Vec<Vec<f32>> = (0..num_rows)
            .map(|_| (0..dimension).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let mut mock_input = MockInput::new(data);

        // Create a temporary directory for output
        let temp_dir = TempDir::new("test_index_writer_process_ivf_hnsw")
            .expect("Failed to create temporary directory");
        let base_directory = temp_dir
            .path()
            .to_str()
            .expect("Failed to convert temporary directory path to string")
            .to_string();

        // Configure IndexWriter
        let base_config = BaseConfig {
            output_path: base_directory.clone(),
            dimension,
            reindex: false,
            max_memory_size: 1024 * 1024 * 1024, // 1 GB
            file_size: 1024 * 1024 * 1024,       // 1 GB
            index_type: IndexType::Spann,
            index_distance_type: DistanceType::L2,
        };
        let quantizer_config = QuantizerConfig {
            quantizer_type: QuantizerType::ProductQuantizer,
            quantizer_distance_type: DistanceType::L2,
            subvector_dimension: 2,
            num_bits: 2,
            num_training_rows: 50,
 
            max_iteration: 10,
            batch_size: 10,
        };
        let hnsw_config = HnswConfig {
            num_layers: 2,
            max_num_neighbors: 10,
            ef_construction: 100,
        };
        let ivf_config = IvfConfig {
            num_clusters: 2,
            num_data_points: 100,
            max_clusters_per_vector: 1,
            distance_threshold: 0.1,

            max_iteration: 10,
            batch_size: 10,
            tolerance: 0.0,
            max_posting_list_size: usize::MAX,
        };
        let config = IndexWriterConfig::Spann(SpannConfigWithBase {
            base_config,
            quantizer_config,
            hnsw_config,
            ivf_config,
        });

        let mut index_writer = IndexWriter::new(config).expect("Failed to create index writer");

        // Process the input
        assert!(index_writer.process(&mut mock_input).is_ok());

        // Check if output directories and files exist
        let spann_directory_path = format!("{}/spann", base_directory);
        let quantizer_directory_path = format!("{}/centroids/quantizer", spann_directory_path);
        let pq_directory = Path::new(&quantizer_directory_path);
        let centroids_directory_path = format!("{}/centroids/hnsw", spann_directory_path);
        let centroids_directory = Path::new(&centroids_directory_path);
        let hnsw_vector_storage_path =
            format!("{}/vector_storage", centroids_directory.to_str().unwrap());
        let hnsw_vector_storage = Path::new(&hnsw_vector_storage_path);
        let hnsw_index_path = format!("{}/index", centroids_directory.to_str().unwrap());
        let hnsw_index = Path::new(&hnsw_index_path);
        assert!(pq_directory.exists());
        assert!(centroids_directory.exists());
        assert!(hnsw_vector_storage.exists());
        assert!(hnsw_index.exists());
    }
}
