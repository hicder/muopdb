#[allow(clippy::module_inception)]
pub mod blocked_bloom_filter;
pub mod bloom_filter;
pub mod builder;
pub mod immutable_bloom_filter;
pub mod writer;

/// Optimized block size: modern processors typically use 64-byte or 128-byte cache lines,
/// so having a block size of 512 bits (64 bytes) would minimize cache misses during lookups.
pub const BLOCK_SIZE_IN_BITS: usize = 512;

/// 3 u64 values:
/// - block_size_in_bits
/// - num_hash_functions
/// - num_blocks
pub const HEADER_SIZE: usize = 24;

#[derive(Copy, Clone)]
pub struct HashIdx {
    // h1 (lower 32 bits of XXH3) selects the block
    // h2 (upper 32 bits of XXH3) sets bits within the block
    //
    // Rationale:
    // - lower bits are more chaotic, reducing collisions between
    // keys mapping to the same block.
    // - upper bits are more stable across hash computations,
    // ensuring uniform bit distribution within a block.
    h1: u32,
    h2: u32,
}
