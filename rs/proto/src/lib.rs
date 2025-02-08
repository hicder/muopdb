pub mod muopdb {
    include!(concat!(env!("OUT_DIR"), "/muopdb.rs"));
    pub const FILE_DESCRIPTOR_SET: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/muopdb_descriptor.bin"));
}

pub mod admin {
    include!(concat!(env!("OUT_DIR"), "/admin.rs"));
}

pub mod aggregator {
    include!(concat!(env!("OUT_DIR"), "/aggregator.rs"));
}
