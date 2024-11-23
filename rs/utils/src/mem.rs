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
