use utils::hdf5_reader::read_hdf5_sift_128;

fn main() {
    let path = "/mnt/dataset/50M_embeddings_128.hdf5";
    match read_hdf5_sift_128(path) {
        Ok(_) => println!("HDF5 file read successfully!"),
        Err(e) => println!("Error reading HDF5 file: {}", e),
    }
}
