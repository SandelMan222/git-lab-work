FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY test/ ./test/

RUN cmake -B build -S test && \
    cmake --build build --parallel