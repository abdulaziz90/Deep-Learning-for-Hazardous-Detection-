# Use NVIDIA's CUDA base image with Ubuntu 20.04 and set CUDA version through an argument
ARG CUDA_VERSION=11.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Update and install required packages
RUN apt-get update && \
    apt-get install -y wget build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev \
    xz-utils tk-dev libffi-dev liblzma-dev python3-openssl git

# Set timezone
RUN echo "Europe/London" > /etc/timezone && dpkg-reconfigure -f noninteractive tzdata

# Install Python 3.10
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar xvf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make altinstall

# Working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip3.10 install --no-cache-dir -r requirements.txt

# Install PyTorch based on CUDA version
COPY install_pytorch.py /app
RUN python3.10 install_pytorch.py

# Copy the code and other required files
COPY . /app

# Default command
CMD ["tail", "-f", "/dev/null"]
