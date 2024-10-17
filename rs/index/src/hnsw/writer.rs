use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
};

use utils::io::wrap_write;

use crate::hnsw::builder::HnswBuilder;

pub struct HnswWriter {
    base_directory: String,
}

pub enum Version {
    V0,
}

pub struct Header {
    pub version: Version,
    pub num_layers: u32,
    pub edges_len: u64,
    pub points_len: u64,
    pub edge_offsets_len: u64,
    pub level_offsets_len: u64,
}

impl HnswWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, index_builder: &HnswBuilder) -> Result<(), String> {
        let non_bottom_layer_nodes = index_builder.get_nodes_from_non_bottom_layer();

        // Edges writer
        let edges_path = format!("{}/edges", self.base_directory);
        let mut edges_file = File::create(edges_path).unwrap();
        let mut edges_buffer_writer = BufWriter::new(&mut edges_file);
        let mut edges_file_len = 0 as u64;

        // Points writer
        let points_path = format!("{}/points", self.base_directory);
        let mut points_file = File::create(points_path).unwrap();
        let mut points_buffer_writer = BufWriter::new(&mut points_file);
        let mut points_file_len = 0 as u64;

        // Edge offsets writer
        let edge_offsets_path = format!("{}/edge_offsets", self.base_directory);
        let mut edge_offsets_file = File::create(edge_offsets_path).unwrap();
        let mut edge_offsets_buffer_writer = BufWriter::new(&mut edge_offsets_file);
        let mut edge_offsets_file_len = 0 as u64;

        // Level offsets writer
        let level_offsets_path = format!("{}/level_offsets", self.base_directory);
        let mut level_offsets_file = File::create(level_offsets_path).unwrap();
        let mut level_offsets_buffer_writer = BufWriter::new(&mut level_offsets_file);
        let mut level_offsets_file_len = 0 as u64;

        let mut current_layer = index_builder.current_top_layer as i32;
        let mut level_offsets = Vec::<usize>::new();
        let mut edge_offsets = Vec::<usize>::new();

        let mut num_edges = 0 as usize;

        // In each level, we append edges and points to file.
        // TODO(hicder): We might not need to write points for level 0...
        while current_layer >= 0 {
            level_offsets.push(edge_offsets.len());

            let mut edges = vec![];
            let mut points = vec![];

            let layer = &index_builder.layers[current_layer as usize];
            if current_layer > 0 {
                for point_id in non_bottom_layer_nodes.iter() {
                    if layer.edges.contains_key(point_id) {
                        // Edge offsets will be the starting index of edges for this point

                        let edges_for_point = &layer.edges[point_id];
                        for edge in edges_for_point {
                            edges.push(edge.point_id);
                        }

                        edge_offsets.push(num_edges);
                        num_edges += edges_for_point.len();

                        points.push(*point_id);
                    }
                }
            } else {
                // Bottom layer, iterate from 0 to max_point_id.
                // Don't need to write points file, since bottom-most layers contain all points
                let max_point_id = *layer.edges.keys().max().unwrap_or(&0);
                for point_id in 0..=max_point_id {
                    edge_offsets.push(num_edges);
                    if layer.edges.contains_key(&point_id) {
                        let edges_for_point = &layer.edges[&point_id];
                        for edge in edges_for_point {
                            edges.push(edge.point_id);
                        }
                        num_edges += edges_for_point.len();
                    }
                }

                // For convienience, after layer 0, just append a level_offset so we can safely do
                // level_offsets[i + 1] - level_offsets[i]
                edge_offsets.push(num_edges);
                level_offsets.push(edge_offsets.len());
            }

            // write edges
            for edge in edges.iter() {
                edges_file_len += wrap_write(&mut edges_buffer_writer, &edge.to_le_bytes())? as u64;
            }

            // write points
            for point in points.iter() {
                points_file_len +=
                    wrap_write(&mut points_buffer_writer, &point.to_le_bytes())? as u64;
            }
            current_layer -= 1;
        }

        // write edge_offsets to file
        for offset in edge_offsets.iter() {
            // In x86_64, usize is the same as u64, but it's safer to use u64 here
            let o = *offset as u64;
            edge_offsets_file_len +=
                wrap_write(&mut edge_offsets_buffer_writer, &o.to_le_bytes())? as u64;
        }

        // write level_offsets to file
        for offset in level_offsets.iter() {
            let o = *offset as u64;
            level_offsets_file_len +=
                wrap_write(&mut level_offsets_buffer_writer, &o.to_le_bytes())? as u64;
        }

        edges_buffer_writer.flush().unwrap();
        points_buffer_writer.flush().unwrap();
        edge_offsets_buffer_writer.flush().unwrap();
        level_offsets_buffer_writer.flush().unwrap();

        let header: Header = Header {
            version: Version::V0,
            num_layers: index_builder.layers.len() as u32,
            edges_len: edges_file_len,
            points_len: points_file_len,
            edge_offsets_len: edge_offsets_file_len,
            level_offsets_len: level_offsets_file_len,
        };

        self.combine_files(header)?;

        // Remove edges, points, edge_offsets, level_offsets. Ok to ignore errors
        fs::remove_file(format!("{}/edges", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/points", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/edge_offsets", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/level_offsets", self.base_directory)).unwrap_or_default();

        Ok(())
    }

    /// Read file and append to the writer
    fn append_file_to_writer(
        &self,
        path: &str,
        writer: &mut BufWriter<&mut File>,
    ) -> Result<(), String> {
        let input_file = File::open(path).unwrap();
        let mut buffer_reader = BufReader::new(&input_file);
        let mut buffer: [u8; 4096] = [0; 4096];
        loop {
            let read = buffer_reader.read(&mut buffer).unwrap();
            wrap_write(writer, &buffer[0..read])?;
            if read < 4096 {
                break;
            }
        }
        Ok(())
    }

    fn write_header(
        &self,
        header: Header,
        writer: &mut BufWriter<&mut File>,
    ) -> Result<(), String> {
        let version_value: u8 = match header.version {
            Version::V0 => 0,
        };
        wrap_write(writer, &version_value.to_le_bytes())?;
        wrap_write(writer, &header.num_layers.to_le_bytes())?;
        wrap_write(writer, &header.edges_len.to_le_bytes())?;
        wrap_write(writer, &header.points_len.to_le_bytes())?;
        wrap_write(writer, &header.edge_offsets_len.to_le_bytes())?;
        wrap_write(writer, &header.level_offsets_len.to_le_bytes())?;
        Ok(())
    }

    /// Combine all individual files into one final index file
    fn combine_files(&self, header: Header) -> Result<(), String> {
        let edges_path = format!("{}/edges", self.base_directory);
        let points_path = format!("{}/points", self.base_directory);
        let edge_offsets_path = format!("{}/edge_offsets", self.base_directory);
        let level_offsets_path = format!("{}/level_offsets", self.base_directory);
        let combined_path = format!("{}/index", self.base_directory);

        let mut combined_file = File::create(combined_path).unwrap();
        let mut combined_buffer_writer = BufWriter::new(&mut combined_file);

        self.write_header(header, &mut combined_buffer_writer)?;
        self.append_file_to_writer(&edges_path, &mut combined_buffer_writer)?;
        self.append_file_to_writer(&points_path, &mut combined_buffer_writer)?;
        self.append_file_to_writer(&edge_offsets_path, &mut combined_buffer_writer)?;
        self.append_file_to_writer(&level_offsets_path, &mut combined_buffer_writer)?;

        match combined_buffer_writer.flush() {
            Ok(_) => Ok(()),
            Err(e) => return Err(e.to_string()),
        }
    }
}

// Test
#[cfg(test)]
mod tests {
    use quantization::{
        pq::{ProductQuantizerConfig, ProductQuantizerWriter},
        pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig},
    };
    use utils::test_utils::generate_random_vector;

    use super::*;

    #[test]
    fn test_write() {
        // Generate 10000 vectors of f32, dimension 128
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        // Create a temporary directory
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let pq_config = ProductQuantizerConfig {
            dimension: 128,
            subvector_dimension: 8,
            num_bits: 8,
            base_directory: base_directory.clone(),
            codebook_name: "codebook".to_string(),
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        // Train a product quantizer
        let pq_writer = ProductQuantizerWriter::new(pq_config.base_directory.clone());
        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        for i in 0..1000 {
            pq_builder.add(datapoints[i].clone());
        }
        let pq = pq_builder.build().unwrap();
        pq_writer.write(&pq).unwrap();

        // Create a HNSW Builder
        let mut hnsw_builder = HnswBuilder::new(10, 128, 20, Box::new(pq));
        for i in 0..datapoints.len() {
            hnsw_builder.insert(i as u64, &datapoints[i]);
        }

        let writer = HnswWriter::new(base_directory.clone());
        match writer.write(&hnsw_builder) {
            Ok(()) => {
                assert!(true);
            }
            Err(_) => {
                assert!(false);
            }
        }
        println!("DONE");
    }
}
