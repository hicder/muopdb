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

pub fn get_ith_val_from_raw_ptr<T: Copy>(raw_ptr: *const T, index: usize) -> T {
    unsafe { *raw_ptr.add(index) }
}

pub struct LowsAndHighs {
    pub lows: Vec<u64>,
    pub highs: Vec<u64>,
}

pub fn u128s_to_lows_highs(ids: &[u128]) -> LowsAndHighs {
    let mut result = LowsAndHighs {
        lows: Vec::with_capacity(ids.len()),
        highs: Vec::with_capacity(ids.len()),
    };

    ids.iter().for_each(|id| {
        result.lows.push(*id as u64);
        result.highs.push((*id >> 64) as u64);
    });

    result
}

pub fn lows_and_highs_to_u128s(lows: &[u64], highs: &[u64]) -> Vec<u128> {
    let mut result = Vec::with_capacity(lows.len());

    lows.iter().zip(highs).for_each(|(low, high)| {
        result.push(*low as u128 | (*high as u128) << 64);
    });

    result
}
