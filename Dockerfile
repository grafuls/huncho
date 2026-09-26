# Syntax=docker/dockerfile:1
#
# huncho serving engine (OPS-02). Builds a slim, single-binary image with the
# ONNX Runtime, Hugging Face Hub, and tokenizers features enabled.
#
# Build (note: the ONNX feature needs network at build time to fetch the
# prebuilt ONNX Runtime via ort's `download-binaries`):
#   docker build -t huncho .
#
# Run (serve a mock model, then POST /v1/systemone with model="mock"):
#   docker run --rm -p 8080:8080 huncho serve --mock --bind 0.0.0.0:8080

# --- builder ---------------------------------------------------------------
FROM rust:1.97-slim-bookworm AS builder

# OpenSSL + pkg-config are required by the `tls-native` feature (ureq -> TLS),
# which backs Hugging Face Hub resolution.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY examples ./examples

# Cache dependency compilation before building our crates.
RUN cargo build --release --locked --features onnx,hf,tokenizers \
    -p huncho-cli

# --- runtime ---------------------------------------------------------------
FROM debian:bookworm-slim AS runtime

# libssl is a dynamic dependency of the HF Hub TLS provider; ca-certificates
# lets the Hub verify the connection at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Drop privileges: the server only needs to read the model cache and bind a port.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin huncho

WORKDIR /app
COPY --from=builder /src/target/release/huncho /usr/local/bin/huncho

# Model cache (OPS-04) is below a writable dir owned by the unprivileged user.
ENV HUNCHO_CACHE_DIR=/var/lib/huncho
RUN mkdir -p "$HUNCHO_CACHE_DIR" && chown -R huncho:huncho "$HUNCHO_CACHE_DIR"

USER huncho
EXPOSE 8080

# Default: serve the built-in mock model on all interfaces so the container is
# useful out of the box. Override with a real manifest/model in your compose
# file or by passing flags.
ENTRYPOINT ["huncho"]
CMD ["serve", "--mock", "--bind", "0.0.0.0:8080"]
