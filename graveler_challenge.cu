#include <curand_kernel.h>
#include <iostream>
#include <chrono>

#define THREADS_PER_BLOCK 256
#define ROLLS_PER_SESSION 231
#define TARGET_COUNT 171

__global__ void simulateRolls(int *max_ones, unsigned long long num_sessions, int seed) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_sessions) return;

    // Initialize random number generator
    curandState state;
    curand_init(seed, idx, 0, &state);

    // Counter for ones
    int count_ones = 0;

    // Simulate a session
    for (int i = 0; i < ROLLS_PER_SESSION; ++i) {
        int roll = curand(&state) % 4; // Random roll between 0 and 3
        if (roll == 0) {
            count_ones++;
        }

        // Early termination if reaching 171 ones is impossible
        if (ROLLS_PER_SESSION - i - 1 + count_ones < TARGET_COUNT) {
            break;
        }
    }

    // Update the global maximum number of ones
    atomicMax(max_ones, count_ones);
}

int main() {
    unsigned long long num_sessions = 1'000'000;
    int h_max_ones = 0;
    int *d_max_ones;

    // Allocate device memory
    cudaMalloc(&d_max_ones, sizeof(int));
    cudaMemcpy(d_max_ones, &h_max_ones, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate number of blocks
    unsigned long long num_blocks = (num_sessions + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    // Launch the kernel
    simulateRolls<<<num_blocks, THREADS_PER_BLOCK>>>(d_max_ones, num_sessions, time(NULL));

    // Synchronize device
    cudaDeviceSynchronize();

    // End timing
    auto end_time = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(&h_max_ones, d_max_ones, sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_max_ones);

    // Calculate the duration
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    // Print result
    std::cout << "Highest Ones Roll: " << h_max_ones << std::endl;
    std::cout << "Number of Roll Sessions: " << num_sessions << std::endl;
    std::cout << "Elapsed Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
}

