# Multi-stage Dockerfile for HRM Vulkan build

# Stage 1: Vulkan SDK and dependencies
FROM ubuntu:22.04 AS base

# Install Vulkan SDK and build tools
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Vulkan SDK
RUN wget -qO - https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - \
    && wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.250-jammy.list \
       https://packages.lunarg.com/vulkan/1.3.250/lunarg-vulkan-1.3.250-jammy.list \
    && apt-get update \
    && apt-get install -y vulkan-sdk \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build HRM
FROM base AS builder

WORKDIR /app

# Copy source
COPY . .

# Build
RUN mkdir build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc)

# Stage 3: Runtime image
FROM ubuntu:22.04

# Install runtime Vulkan libraries
RUN apt-get update && apt-get install -y \
    libvulkan1 \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy built binaries
COPY --from=builder /app/build/release/ /app/

# Copy assets and shaders
COPY --from=builder /app/assets/ /app/assets/
COPY --from=builder /app/shaders/ /app/shaders/

WORKDIR /app

# Default command
CMD ["./hrm_system", "--cli"]