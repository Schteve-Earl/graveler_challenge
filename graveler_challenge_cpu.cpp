#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip> // Include this header for std::fixed and std::setprecision

#define NUM_ITEMS 4
#define NUM_ROLLS 231
#define NUM_SIMULATIONS 1000000000 // 1,000,000,000
#define NUM_THREADS 8 // Adjust based on your CPU's number of cores

std::mutex maxOnesMutex;
int maxOnes = 0;

void simulateRolls(int threadId, int simulationsPerThread, std::mt19937 &gen, std::uniform_int_distribution<int> &dist) {
    for (int i = 0; i < simulationsPerThread; ++i) {
        int counts[NUM_ITEMS] = {0};

        for (int j = 0; j < NUM_ROLLS; ++j) {
            int roll = dist(gen);
            counts[roll]++;
        }

        int ones = counts[0];
        
        std::lock_guard<std::mutex> guard(maxOnesMutex);
        if (ones > maxOnes) {
            maxOnes = ones;
        }
    }
}

int main() {
    int simulationsPerThread = NUM_SIMULATIONS / NUM_THREADS;
    std::vector<std::thread> threads;
    std::random_device rd;
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < NUM_THREADS; ++t) {
        std::mt19937 gen(rd() + t); // Seed with different values for each thread
        std::uniform_int_distribution<int> dist(0, NUM_ITEMS - 1);
        threads.emplace_back(simulateRolls, t, simulationsPerThread, std::ref(gen), std::ref(dist));
    }

    for (auto &thread : threads) {
        thread.join();
    }

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Highest Ones Roll: " << maxOnes << std::endl;
    std::cout << "Execution time: " << std::fixed << std::setprecision(6) << elapsed.count() << " seconds" << std::endl;

    return 0;
}

