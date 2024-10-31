pub mod hdf5;

pub struct Row<'a> {
    pub id: u64,
    pub data: &'a [f32],
}

pub trait Input {
    // Return true if there are more rows to read
    fn has_next(&self) -> bool;

    // Return the next row of data
    fn next(&mut self) -> Row;

    // Reset the state of the input to the beginning
    // This is helpful when we want to do multiple passes over the same input
    fn reset(&mut self);

    // Return the number of rows in the input
    fn num_rows(&self) -> usize;
}
