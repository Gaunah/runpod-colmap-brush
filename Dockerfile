ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

# ==========================================
# Stage 1: Build COLMAP (with Caspar GPU bundle adjustment)
# ==========================================
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS colmap-builder
ENV DEBIAN_FRONTEND=noninteractive
ARG COLMAP_REF=4.2.0
# Caspar requires compute capability >= 7.5, so "all-major" (which includes
# sm_50/60/70) can't be used. Turing, Ampere, Ada, Hopper, Blackwell.
ARG CUDA_ARCHITECTURES="75;80;86;89;90;120"

# Install build dependencies
RUN apt update && \
    apt install -y git ccache cmake ninja-build build-essential \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev \
        libopenimageio-dev openimageio-tools libmetis-dev libgoogle-glog-dev libgtest-dev \
        libgmock-dev libsqlite3-dev libglew-dev qt6-base-dev libqt6opengl6-dev \
        libqt6openglwidgets6 libqt6svg6-dev libcgal-dev libceres-dev libcurl4-openssl-dev \
        libssl-dev libmkl-full-dev

RUN mkdir -p /usr/include/opencv4

# Shallow clone and build COLMAP. Caspar is vendored in COLMAP's thirdparty
# tree, so CASPAR_ENABLED needs no extra dependencies beyond CUDA.
RUN git clone --depth 1 --branch ${COLMAP_REF} https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    mkdir -p build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
        -DCASPAR_ENABLED=ON \
        -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DBLA_VENDOR=Intel10_64lp && \
    ninja install

# ==========================================
# Stage 2: Build Brush
# ==========================================
FROM rust AS brush-builder
RUN apt update && apt install -y git

# Shallow clone and build Brush
RUN git clone --depth 1 https://github.com/ArthurBrussee/brush.git /build
WORKDIR /build
RUN cargo build --release -p brush-app

# ==========================================
# Stage 3: Final RunPod Runtime (With JupyterLab)
# ==========================================
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 1. Runtime dependencies
RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 libc6 libomp5 libopengl0 libmetis5 \
        libceres4t64 libopenimageio2.4t64 libgcc-s1 libgl1 libglew2.2 \
        libgoogle-glog0v6t64 libqt6core6 libqt6gui6 libqt6widgets6 \
        libqt6openglwidgets6 libqt6svg6 libcurl4 libssl3t64 \
        libmkl-locale libmkl-intel-lp64 libmkl-intel-thread libmkl-core \
        libvulkan1 mesa-vulkan-drivers \
        python3 python3-pip wget 7zip tmux neovim curl aria2 sqlite3 npm && \
    npm install -g @playcanvas/splat-transform && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# 2. JupyterLab + Python deps used by run_pipeline.sh (plyfile/numpy for
#    the NaN-splat cleanup step before SOG compression)
RUN pip3 install jupyterlab plyfile numpy --break-system-packages --no-cache-dir

# 3. Copy binaries from builders
COPY --from=colmap-builder /colmap-install/ /usr/local/
COPY --from=brush-builder /build/target/release/brush /usr/local/bin/brush

# 3b. Fail the build immediately if COLMAP has unresolved shared libs
RUN ldd /usr/local/bin/colmap | grep "not found" && exit 1 || true

# 4. Set up RunPod Workspace
WORKDIR /workspace

# 5. Download the Vocab Tree
RUN mkdir /app
RUN wget https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin -P /app/

# 6. Pipeline + Jupyter startup scripts
COPY run_pipeline.sh /app/run_pipeline.sh
RUN chmod +x /app/run_pipeline.sh
COPY start-jupyter.sh /app/start-jupyter.sh
RUN chmod +x /app/start-jupyter.sh

# 7. Expose Jupyter's default port
EXPOSE 8888

# 8. Start JupyterLab
CMD ["/app/start-jupyter.sh"]
