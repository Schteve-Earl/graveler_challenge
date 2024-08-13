# Start from the NVIDIA CUDA image
FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the source file into the container
COPY graveler_challenge.cu /app/

# Build the CUDA program
RUN nvcc -o graveler_challenge graveler_challenge.cu

# Set the entrypoint to run the program
ENTRYPOINT ["./graveler_challenge"]

