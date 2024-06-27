FROM rust:1.71 AS rust_base
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS cuda_base
FROM registry-wcsng.nrp-nautilus.io/gnuradio-devel:v3.10.9.1_4.6.0.0 AS uhd_base

FROM cuda_base AS merged_base
COPY --from=uhd_base /usr/local /usr/local
COPY --from=rust_base /usr/local/cargo /usr/local/cargo
COPY --from=rust_base /usr/local/rustup /usr/local/rustup

ENV PATH="/usr/local/cargo/bin:${PATH}"
ENV RUSTUP_HOME="/usr/local/rustup"
ENV CARGO_HOME="/usr/local/cargo"
ENV DEBIAN_FRONTEND noninteractive

RUN rustup component add rustfmt

RUN apt-get update && \
    apt-get install -y lsb-release 

RUN apt-get update && apt-get -y install cmake 

RUN apt-get update && apt-get install -y python3.10 python3-pip

RUN apt-get update && apt-get install -y vim wget sudo software-properties-common gnupg git 

RUN apt-get update && apt-get install ninja-build

RUN apt-get update && apt-get install -y cuda-nsight-systems-11-8

RUN apt-get update && apt-get install -y zip 

# For UHD
RUN apt-get update && apt-get -y install \
    pkg-config \
    clang \
    libboost-all-dev \
    libusb-1.0-0-dev

RUN pip install numpy scipy matplotlib

COPY . /root/channelizer

# Set user specific environment variables
ENV USER root
ENV HOME /root
# Switch to user
USER ${USER}
