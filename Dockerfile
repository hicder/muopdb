FROM ubuntu:24.04 AS builder

# Install rust
RUN apt-get update
RUN apt-get install -y curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install hdf5 protobuf openblas clang
RUN apt-get install -y libhdf5-dev libprotobuf-dev libopenblas-dev clang protobuf-compiler pkg-config

# Install nightly toolchain
RUN rustup install nightly-x86_64-unknown-linux-gnu && \
    rustup default nightly-x86_64-unknown-linux-gnu

# Copy stuff
COPY . /muopdb
WORKDIR /muopdb
RUN cargo build --release

# Test
FROM builder AS test
RUN cargo test --release --features async-hnsw

# Index Server
FROM builder AS index_server

RUN mkdir /app
RUN mv /muopdb/target/release/index_server /app/index_server

RUN rm -rf /muopdb

CMD ["/app/index_server", "--node-id", "0", "--index-config-path", "/mnt/muopdb/indices", "--index-data-path", "/mnt/muopdb/data", "--port", "9002"]
