use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_file = "./proto/muopdb.proto";
    let admin_proto_file = "./proto/admin.proto";
    let aggregator_proto_file = "./proto/aggregator.proto";
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);

    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional") // for older systems
        .build_client(true)
        .build_server(true)
        .file_descriptor_set_path(out_dir.join("muopdb_descriptor.bin"))
        .out_dir(out_dir.clone())
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile_protos(&[proto_file, admin_proto_file], &["proto"])?;

    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional") // for older systems
        .build_client(true)
        .build_server(true)
        .file_descriptor_set_path(out_dir.join("aggregator_descriptor.bin"))
        .out_dir(out_dir)
        .type_attribute(".", "#[derive(serde::Serialize, serde::Deserialize)]")
        .compile_protos(&[aggregator_proto_file], &["proto"])?;

    let output = Command::new("cargo")
        .args(&["fmt"])
        .output()
        .expect("Failed to execute cargo fmt");

    if !output.status.success() {
        panic!(
            "cargo fmt failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(())
}
