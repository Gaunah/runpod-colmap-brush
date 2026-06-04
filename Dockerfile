ARG UBUNTU_VERSION=24.04
ARG NVIDIA_CUDA_VERSION=12.9.1

# ==========================================
# Stage 1: Build COLMAP (with CUDA+cuDSS Ceres)
# ==========================================
FROM nvidia/cuda:${NVIDIA_CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS colmap-builder
ENV DEBIAN_FRONTEND=noninteractive
ARG COLMAP_REF=4.0.4
ARG CUDSS_VERSION=0.7.1

# Build dependencies (NO libceres-dev — we build Ceres ourselves;
# suitesparse + gflags added for the custom Ceres build)
RUN apt update && \
    apt install -y git ccache cmake ninja-build build-essential wget \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev libeigen3-dev \
        libopenimageio-dev openimageio-tools libmetis-dev libgoogle-glog-dev libgflags-dev \
        libgtest-dev libgmock-dev libsqlite3-dev libglew-dev qt6-base-dev libqt6opengl6-dev \
        libqt6openglwidgets6 libqt6svg6-dev libcgal-dev libcurl4-openssl-dev \
        libssl-dev libmkl-full-dev libsuitesparse-dev

# Pin cuDSS to ${CUDSS_VERSION} — the CUDA network repo carries multiple
# versions and apt would otherwise pick the newest (0.8.x), whose API
# breaks the Ceres build. Pin-Priority 1001 wins over any version number.
RUN printf 'Package: *cudss*\nPin: version %s*\nPin-Priority: 1001\n' "${CUDSS_VERSION}" \
        > /etc/apt/preferences.d/99-cudss-pin && \
    apt update && \
    apt install -y cudss-cuda-12 libcudss0-cuda-12 libcudss0-dev-cuda-12 libcudss0-static-cuda-12 && \
    dpkg -l | grep cudss && \
    dpkg -l | grep -q "libcudss0-cuda-12.*${CUDSS_VERSION}" || \
        (echo "ERROR: wrong cuDSS version installed" && exit 1)

# Build Ceres from source with CUDA + cuDSS.
# NOTE: unpinned master (identifies as 2.3.0-pre); no release tag with cuDSS
# support exists yet. Pin a commit SHA here once you have a known-good build.
RUN git clone --depth 1 --recursive --shallow-submodules https://github.com/ceres-solver/ceres-solver.git /ceres && \
    cd /ceres && mkdir build && cd build && \
    cmake .. -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DUSE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=all-major \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF && \
    ninja install

RUN mkdir -p /usr/include/opencv4

# Build COLMAP — picks up the custom Ceres from /usr/local automatically
RUN git clone --depth 1 --branch ${COLMAP_REF} https://github.com/colmap/colmap.git /colmap && \
    cd /colmap && \
    mkdir -p build && cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=all-major -DCMAKE_INSTALL_PREFIX=/colmap-install -DBLA_VENDOR=Intel10_64lp && \
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
ARG CUDSS_VERSION=0.7.1

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 1. Runtime dependencies.
#    - libceres4t64 REMOVED: Ceres is now statically linked into COLMAP.
#    - SuiteSparse runtime libs ADDED: the custom Ceres links them
#      dynamically (apt's libceres used to pull these in for us).
RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends --no-install-suggests \
        libboost-program-options1.83.0 libc6 libomp5 libopengl0 libmetis5 \
        libopenimageio2.4t64 libgcc-s1 libgl1 libglew2.2 \
        libgoogle-glog0v6t64 libqt6core6 libqt6gui6 libqt6widgets6 \
        libqt6openglwidgets6 libqt6svg6 libcurl4 libssl3t64 \
        libmkl-locale libmkl-intel-lp64 libmkl-intel-thread libmkl-core \
        libcholmod5 libspqr4 libamd3 libcamd3 libccolamd3 libcolamd3 \
        libcxsparse4 libsuitesparseconfig7 \
        libvulkan1 mesa-vulkan-drivers \
        python3 python3-pip wget 7zip tmux neovim curl aria2 sqlite3 npm && \
    npm install -g @playcanvas/splat-transform && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# 1b. cuDSS runtime — SAME pinned version as the builder, or COLMAP's
#     dynamically linked libcudss.so.0 won't match what it was built against.
RUN printf 'Package: *cudss*\nPin: version %s*\nPin-Priority: 1001\n' "${CUDSS_VERSION}" \
        > /etc/apt/preferences.d/99-cudss-pin && \
    apt update && \
    apt install -y --no-install-recommends cudss-cuda-12 libcudss0-cuda-12 && \
    dpkg -l | grep -q "libcudss0-cuda-12.*${CUDSS_VERSION}" || \
        (echo "ERROR: wrong cuDSS version installed" && exit 1) && \
    apt clean && rm -rf /var/lib/apt/lists/*

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
