use num_traits::ops::bytes::ToBytes;

pub mod file;

/// Trait that defines the interface for vector storage
/// This storage owns the actual vector, and will return a reference to it
pub trait VectorStorage<T: ToBytes + Clone> {
    fn get(&self, id: u32) -> Option<&[T]>;
    fn append(&mut self, vector: Vec<T>) -> Result<(), String>;
}
