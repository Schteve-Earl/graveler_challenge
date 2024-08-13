#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <iomanip>  // For std::fixed and std::setprecision
#include <chrono>

#define NUM_ITEMS 4
#define NUM_ROLLS 231
#define THREADS_PER_BLOCK 256

__global__ void simulateRolls(int *maxOnes, long long numSimulations, int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long step = gridDim.x * blockDim.x;

    // Initialize random state for each thread
    curandState state;
    curand_init(seed, idx, 0, &state);

    for (long long i = idx; i < numSimulations; i += step) {
        int counts[NUM_ITEMS] = {0, 0, 0, 0};

        for (int j = 0; j < NUM_ROLLS; j++) {
            int roll = curand(&state) % NUM_ITEMS; // Get a random number in [0, NUM_ITEMS)
            counts[roll]++;
            
            // Early exit if the remaining rolls cannot reach a new max
            if (counts[0] + (NUM_ROLLS - j - 1) < *maxOnes) {
                break;
            }
        }

        // Update the maximum number of ones rolled
        atomicMax(maxOnes, counts[0]);
    }
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Start total execution timing
    auto total_start = std::chrono::high_resolution_clock::now();

    long long numSimulations = 1000000000; // 1,000,000,000
    int *d_maxOnes, h_maxOnes = 0;

    // Query device properties to determine max blocks
    cudaDeviceProp prop;
    checkCuda(cudaGetDeviceProperties(&prop, 0));

    int maxThreadsPerBlock = prop.maxThreadsPerBlock;
    int maxBlocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK);
    std::cout << "Launching with " << maxBlocks << " blocks and " << THREADS_PER_BLOCK << " threads per block." << std::endl;

    checkCuda(cudaMalloc(&d_maxOnes, sizeof(int)));
    checkCuda(cudaMemset(d_maxOnes, 0, sizeof(int)));

    // Start kernel execution timing
    auto kernel_start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    simulateRolls<<<maxBlocks, THREADS_PER_BLOCK>>>(d_maxOnes, numSimulations, std::rand());
    checkCuda(cudaDeviceSynchronize());

    // End kernel execution timing
    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kernel_elapsed = kernel_end - kernel_start;

    checkCuda(cudaMemcpy(&h_maxOnes, d_maxOnes, sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(d_maxOnes);

    // End total execution timing
    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = total_end - total_start;

    std::cout << "Highest Ones Roll: " << h_maxOnes << std::endl;
    std::cout << "Kernel execution time: " << std::fixed << std::setprecision(6) << kernel_elapsed.count() << " seconds" << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(6) << total_elapsed.count() << " seconds" << std::endl;

    return 0;
}

