use std::cmp::min;
use std::path::Path;

use hdf5::File;
use ndarray::{s, Array2};

/// Sample function to read a HDF5 file
/// TODO(hicder): Fix this function to make it generic
pub fn read_hdf5_sift_128(path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = Path::new(path);
    let file = File::open(file_path)?;
    let datasets = file.datasets().unwrap();
    for dataset in datasets {
        let name = dataset.name();
        println!("Dataset: {}", name);

        if name == "/train" {
            let res = dataset.read_2d::<f32>().unwrap();
            // let mut train_data = Vec::<(Vec<f32>, usize)>::new();
            let (nrows, ncols) = res.dim();
            println!("Rows: {}, Cols: {}", nrows, ncols);

            // Print the first 20 rows
            for i in 0..20 {
                for j in 0..10 {
                    print!("{:?},", res.get((i, j)).unwrap());
                }
                println!()
            }
        } else if name == "/neighbors" {
            let res = dataset.read_2d::<f32>().unwrap();
            let (nrows, ncols) = res.dim();
            println!("Rows: {}, Cols: {}", nrows, ncols);

            for i in 0..20 {
                for j in 0..10 {
                    print!("{:?},", res.get((i, j)).unwrap());
                }
                println!();
            }
        } else {
            let nrows = dataset.shape()[0];
            let ncols = dataset.shape()[1];

            let chunk_size = 100;
            let i = 0;
            let selection = s![i..min(i + chunk_size, nrows), ..];
            let chunk: Array2<f32> = dataset.read_slice_2d(selection).unwrap();
            for row in chunk.axis_iter(ndarray::Axis(0)) {
                let v = row.to_vec();
                println!("{:?}", v);
            }
            println!("Rows: {}, Cols: {}", nrows, ncols);
        }
    }

    println!("HDF5 file opened successfully!");
    Ok(())
}
