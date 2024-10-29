use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Input file
    #[arg(short, long)]
    input_path: String,

    #[arg(short, long)]
    output_path: String,
}

fn main() {
    let arg = Args::parse();

    println!("{:?}", arg);
}
