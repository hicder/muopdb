use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
   let proto_file = "./proto/muopdb.proto";
   let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

   tonic_build::configure()
           .protoc_arg("--experimental_allow_proto3_optional") // for older systems
           .build_client(true)
           .build_server(true)
           .file_descriptor_set_path(out_dir.join("aggregator_descriptor.bin"))
           .out_dir("./src")
           .compile(&[proto_file], &["proto"])?;

   Ok(())
}
