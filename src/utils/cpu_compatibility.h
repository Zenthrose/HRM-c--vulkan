#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

enum class CpuVendor {
    Intel,
    AMD,
    ARM,
    Apple,  // For Apple Silicon
    Unknown
};

enum class CpuArchitecture {
    x86_64,
    ARM64,
    Unknown
};

struct CpuFeatures {
    bool has_sse = false;
    bool has_sse2 = false;
    bool has_sse3 = false;
    bool has_ssse3 = false;
    bool has_sse4_1 = false;
    bool has_sse4_2 = false;
    bool has_avx = false;
    bool has_avx2 = false;
    bool has_avx512 = false;
    bool has_fma = false;
    bool has_neon = false;  // ARM
    int num_cores = 1;
    int num_threads = 1;
};

class CpuCompatibility {
public:
    static CpuFeatures detectCpuFeatures();
    static CpuVendor detectCpuVendor();
    static CpuArchitecture detectCpuArchitecture();
    static bool isLittleEndian();
    static void* alignedAlloc(size_t size, size_t alignment);
    static void alignedFree(void* ptr);
    static std::string getCpuInfoString();

    // SIMD-accelerated operations
    static float vectorSum(const float* data, size_t size);
    static void vectorAdd(float* result, const float* a, const float* b, size_t size);
    static void vectorScale(float* result, const float* data, float scale, size_t size);

    // Endianness handling
    static uint32_t swapEndian32(uint32_t value);
    static uint64_t swapEndian64(uint64_t value);
    static void swapEndian(float* data, size_t size);
    static bool shouldSwapEndianForFile(); // For cross-platform file compatibility

private:
    static CpuFeatures detectX86Features();
    static CpuFeatures detectArmFeatures();
};