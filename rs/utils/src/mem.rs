pub fn transmute_u8_to_slice<T>(data: &[u8]) -> &[T] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const T,
            data.len() / std::mem::size_of::<T>(),
        )
    }
}

pub fn transmute_u8_to_val<T: Copy>(data: &[u8]) -> T {
    unsafe { *(data.as_ptr() as *const T) }
}

pub fn transmute_slice_to_u8<T>(slice: &[T]) -> &[u8] {
    let byte_count = slice.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, byte_count) }
}
