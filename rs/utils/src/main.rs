use utils::hdf5_reader::read_hdf5_sift_128;

fn main() {
    let path = "/home/hieu/Downloads/sift-128-euclidean.hdf5";
    match read_hdf5_sift_128(path) {
        Ok(_) => println!("HDF5 file read successfully!"),
        Err(e) => println!("Error reading HDF5 file: {}", e),
    }
}
