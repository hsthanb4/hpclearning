# 使用官方 CUDA 12.2 开发版作为基准
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# 镜像元数据设置
LABEL maintainer="hsthanb4's GPU Dev Env"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    wget \
    git \
    vim \
    locales \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 还原镜像中 2 months ago 增加的 CMake 3.28.0 手动安装步骤
# 使用了你在 history 中看到的 CMAKE_VERSION=3.28.0
ARG CMAKE_VERSION=3.28.0
ARG NUM_JOBS=8
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz \
    && tar -zxvf cmake-${CMAKE_VERSION}.tar.gz \
    && cd cmake-${CMAKE_VERSION} \
    && ./bootstrap \
    && make -j${NUM_JOBS} \
    && make install \
    && cd .. && rm -rf cmake-${CMAKE_VERSION}*

# 设置工作目录
WORKDIR /workspace

# 默认进入 bash
CMD ["/bin/bash"]