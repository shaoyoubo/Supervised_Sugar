#!/bin/bash
set -e

echo "Updating package lists..."
apt-get update

echo "Installing COLMAP dependencies (excluding nvidia-cuda-toolkit)..."
apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libgmock-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev

echo "Cloning COLMAP source..."
cd ~
git clone https://github.com/colmap/colmap.git

cd colmap

mkdir -p build
cd build

echo "Configuring COLMAP with CMake (using your installed CUDA 11.8)..."
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86"

echo "Building COLMAP..."
ninja

echo "Installing COLMAP..."
ninja install

echo "COLMAP with CUDA 11.8 support installed successfully!"
