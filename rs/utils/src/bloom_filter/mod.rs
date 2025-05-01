#[allow(clippy::module_inception)]
pub mod blocked_bloom_filter;
pub mod bloom_filter;
pub mod builder;
pub mod writer;

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
