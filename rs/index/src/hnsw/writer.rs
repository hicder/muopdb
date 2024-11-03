use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};

use anyhow::Context;
use utils::io::wrap_write;

use crate::hnsw::builder::HnswBuilder;

pub struct HnswWriter {
    base_directory: String,
}

#[derive(PartialEq, Debug)]
pub enum Version {
    V0,
}

#[derive(Debug)]
pub struct Header {
    pub version: Version,
    pub quantized_dimension: u32,
    pub num_layers: u32,
    pub edges_len: u64,
    pub points_len: u64,
    pub edge_offsets_len: u64,
    pub level_offsets_len: u64,
    pub doc_id_mapping_len: u64,
}

impl HnswWriter {
    pub fn new(base_directory: String) -> Self {
        Self { base_directory }
    }

    pub fn write(&self, index_builder: &mut HnswBuilder, reindex: bool) -> anyhow::Result<()> {
        let non_bottom_layer_nodes = index_builder.get_nodes_from_non_bottom_layer();
        if reindex {
            // Reindex the HNSW to efficient lookup
            index_builder
                .reindex()
                .context("failed to reindex during write")?;
        }
        // Doc_id mapping writer
        let doc_id_mapping_path = format!("{}/doc_id_mapping", self.base_directory);
        let mut doc_id_mapping_file = File::create(doc_id_mapping_path).unwrap();
        let mut doc_id_mapping_buffer_writer = BufWriter::new(&mut doc_id_mapping_file);
        let mut doc_id_mapping_file_len = 0 as u64;

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

        let vectors_path = format!("{}/vector_storage", self.base_directory);
        let mut vectors_file = File::create(vectors_path).unwrap();
        let mut vectors_buffer_writer = BufWriter::new(&mut vectors_file);
        index_builder.vectors().write(&mut vectors_buffer_writer)?;

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

        // write doc_id_mapping to file, id_mapping should be from 0 to n-1 where n is the number of points
        for doc_id in &index_builder.doc_id_mapping {
            doc_id_mapping_file_len +=
                wrap_write(&mut doc_id_mapping_buffer_writer, &doc_id.to_le_bytes())? as u64;
        }

        edges_buffer_writer.flush().unwrap();
        points_buffer_writer.flush().unwrap();
        edge_offsets_buffer_writer.flush().unwrap();
        level_offsets_buffer_writer.flush().unwrap();
        doc_id_mapping_buffer_writer.flush().unwrap();

        let header: Header = Header {
            version: Version::V0,
            quantized_dimension: index_builder.quantizer.quantized_dimension() as u32,
            num_layers: index_builder.layers.len() as u32,
            edges_len: edges_file_len,
            points_len: points_file_len,
            edge_offsets_len: edge_offsets_file_len,
            level_offsets_len: level_offsets_file_len,
            doc_id_mapping_len: doc_id_mapping_file_len,
        };

        self.combine_files(header)?;

        // Remove edges, points, edge_offsets, level_offsets. Ok to ignore errors
        fs::remove_file(format!("{}/edges", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/points", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/edge_offsets", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/level_offsets", self.base_directory)).unwrap_or_default();
        fs::remove_file(format!("{}/doc_id_mapping", self.base_directory)).unwrap_or_default();

        Ok(())
    }

    /// Read file and append to the writer
    fn append_file_to_writer(
        &self,
        path: &str,
        writer: &mut BufWriter<&mut File>,
    ) -> anyhow::Result<usize> {
        let input_file = File::open(path).unwrap();
        let mut buffer_reader = BufReader::new(&input_file);
        let mut buffer: [u8; 4096] = [0; 4096];
        let mut written = 0;
        loop {
            let read = buffer_reader.read(&mut buffer).unwrap();
            written += wrap_write(writer, &buffer[0..read])?;
            if read < 4096 {
                break;
            }
        }
        Ok(written)
    }

    fn write_header(
        &self,
        header: Header,
        writer: &mut BufWriter<&mut File>,
    ) -> anyhow::Result<usize> {
        let version_value: u8 = match header.version {
            Version::V0 => 0,
        };
        let mut written = 0;
        written += wrap_write(writer, &version_value.to_le_bytes())?;
        written += wrap_write(writer, &header.quantized_dimension.to_le_bytes())?;
        written += wrap_write(writer, &header.num_layers.to_le_bytes())?;
        written += wrap_write(writer, &header.edges_len.to_le_bytes())?;
        written += wrap_write(writer, &header.points_len.to_le_bytes())?;
        written += wrap_write(writer, &header.edge_offsets_len.to_le_bytes())?;
        written += wrap_write(writer, &header.level_offsets_len.to_le_bytes())?;
        written += wrap_write(writer, &header.doc_id_mapping_len.to_le_bytes())?;
        Ok(written)
    }

    /// Combine all individual files into one final index file
    fn combine_files(&self, header: Header) -> anyhow::Result<usize> {
        let edges_path = format!("{}/edges", self.base_directory);
        let points_path = format!("{}/points", self.base_directory);
        let edge_offsets_path = format!("{}/edge_offsets", self.base_directory);
        let level_offsets_path = format!("{}/level_offsets", self.base_directory);
        let doc_id_mapping_path = format!("{}/doc_id_mapping", self.base_directory);
        let combined_path = format!("{}/index", self.base_directory);

        let mut combined_file = File::create(combined_path).unwrap();
        let mut combined_buffer_writer = BufWriter::new(&mut combined_file);

        let mut written = self
            .write_header(header, &mut combined_buffer_writer)
            .context("failed to write header")?;
        // Compute pading for alignment to 4 bytes
        let mut padding = 4 - (written % 4);
        if padding != 4 {
            let padding_buffer = vec![0; padding];
            written += wrap_write(&mut combined_buffer_writer, &padding_buffer)?;
        }
        written += self.append_file_to_writer(&edges_path, &mut combined_buffer_writer)?;
        written += self.append_file_to_writer(&points_path, &mut combined_buffer_writer)?;

        padding = 8 - (written % 8);
        if padding != 8 {
            let padding_buffer = vec![0; padding];
            written += wrap_write(&mut combined_buffer_writer, &padding_buffer)?;
        }
        written += self.append_file_to_writer(&edge_offsets_path, &mut combined_buffer_writer)?;
        written += self.append_file_to_writer(&level_offsets_path, &mut combined_buffer_writer)?;
        written += self.append_file_to_writer(&doc_id_mapping_path, &mut combined_buffer_writer)?;

        combined_buffer_writer
            .flush()
            .context("failed to flush combined buffer")?;
        Ok(written)
    }
}

// Test
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::vec;

    use ordered_float::NotNan;
    use quantization::pq::{ProductQuantizerConfig, ProductQuantizerWriter};
    use quantization::pq_builder::{ProductQuantizerBuilder, ProductQuantizerBuilderConfig};
    use utils::test_utils::generate_random_vector;

    use super::*;
    use crate::hnsw::builder::Layer;
    use crate::hnsw::reader::HnswReader;
    use crate::hnsw::utils::{GraphTraversal, PointAndDistance};
    use crate::vector::file::FileBackedVectorStorage;

    fn construct_layers(hnsw_builder: &mut HnswBuilder) {
        // Prepare all layers
        for _ in 0..3 {
            hnsw_builder.layers.push(Layer {
                edges: HashMap::new(),
            });
        }
        // layer 2
        {
            let layer = hnsw_builder.layers.get_mut(2).unwrap();
            layer.edges.insert(1, vec![]);
        }
        // Layer 1
        {
            let layer = hnsw_builder.layers.get_mut(1).unwrap();
            layer.edges.insert(
                1,
                vec![
                    PointAndDistance {
                        point_id: 4,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 5,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                4,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 5,
                        distance: NotNan::new(3.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                5,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 4,
                        distance: NotNan::new(3.0).unwrap(),
                    },
                ],
            );
        }

        // layer 0
        {
            let layer = hnsw_builder.layers.get_mut(0).unwrap();
            layer.edges.insert(
                1,
                vec![
                    PointAndDistance {
                        point_id: 4,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 5,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                4,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 5,
                        distance: NotNan::new(3.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                5,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 4,
                        distance: NotNan::new(3.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                2,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 3,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                3,
                vec![
                    PointAndDistance {
                        point_id: 2,
                        distance: NotNan::new(1.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 4,
                        distance: NotNan::new(2.0).unwrap(),
                    },
                ],
            );
            layer.edges.insert(
                0,
                vec![
                    PointAndDistance {
                        point_id: 1,
                        distance: NotNan::new(3.0).unwrap(),
                    },
                    PointAndDistance {
                        point_id: 2,
                        distance: NotNan::new(4.0).unwrap(),
                    },
                ],
            );
        }
    }

    #[test]
    fn test_write() {
        // Generate 10000 vectors of f32, dimension 128
        let datapoints: Vec<Vec<f32>> = (0..10000).map(|_| generate_random_vector(128)).collect();

        // Create a temporary directory
        let temp_dir = tempdir::TempDir::new("product_quantizer_test").unwrap();
        let base_directory = temp_dir.path().to_str().unwrap().to_string();
        let vector_dir = format!("{}/vectors", base_directory);
        fs::create_dir_all(vector_dir.clone()).unwrap();
        let vectors = Box::new(FileBackedVectorStorage::<u8>::new(
            vector_dir, 1024, 4096, 16,
        ));
        let pq_config = ProductQuantizerConfig {
            dimension: 128,
            subvector_dimension: 8,
            num_bits: 8,
        };

        let pq_builder_config = ProductQuantizerBuilderConfig {
            max_iteration: 1000,
            batch_size: 4,
        };

        // Train a product quantizer
        let pq_writer = ProductQuantizerWriter::new(base_directory.clone());
        let mut pq_builder = ProductQuantizerBuilder::new(pq_config, pq_builder_config);

        for i in 0..1000 {
            pq_builder.add(datapoints[i].clone());
        }
        let pq = pq_builder.build(base_directory.clone()).unwrap();
        pq_writer.write(&pq).unwrap();

        // Create a HNSW Builder
        let mut hnsw_builder = HnswBuilder::new(10, 128, 20, Box::new(pq), vectors);
        hnsw_builder
            .vectors()
            .append(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        hnsw_builder
            .vectors()
            .append(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        hnsw_builder
            .vectors()
            .append(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        hnsw_builder
            .vectors()
            .append(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        hnsw_builder
            .vectors()
            .append(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        hnsw_builder
            .vectors()
            .append(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();

        // Artificially construct the graph, since inserting is not deterministic.
        // 3 layers: 0 to 2
        hnsw_builder.doc_id_mapping = vec![1, 2, 3, 4, 5, 6];
        hnsw_builder.current_top_layer = 2;
        construct_layers(&mut hnsw_builder);
        hnsw_builder.entry_point = vec![1];

        // Write to disk
        let writer = HnswWriter::new(base_directory.clone());

        writer.write(&mut hnsw_builder, false).unwrap();

        let reader = HnswReader::new(base_directory.clone());
        let hnsw = reader.read();
        {
            let egdes = hnsw.get_edges_for_point(1, 2);
            assert!(egdes.is_none());
        }
        {
            let edges = hnsw.get_edges_for_point(1, 1).unwrap();
            assert_eq!(edges.len(), 2);
            assert!(edges.contains(&4));
            assert!(edges.contains(&5));
        }
        {
            let edges = hnsw.get_edges_for_point(0, 0).unwrap();
            assert_eq!(edges.len(), 2);
            assert!(edges.contains(&1));
            assert!(edges.contains(&2));
        }

        assert_eq!(
            hnsw.get_doc_id_test(&[0, 1, 2, 3, 4, 5]),
            vec![1, 2, 3, 4, 5, 6]
        );

        assert_eq!(hnsw.get_header().version, Version::V0);
        assert_eq!(hnsw.get_header().num_layers, 3);
    }
}
