#include "cpu_compatibility.h"
#include <iostream>
#include <thread>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cstdint>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

#if defined(__AVX__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

CpuFeatures CpuCompatibility::detectCpuFeatures() {
    CpuFeatures features;

    features.num_cores = std::thread::hardware_concurrency();
    features.num_threads = std::thread::hardware_concurrency();

    CpuArchitecture arch = detectCpuArchitecture();
    if (arch == CpuArchitecture::x86_64) {
        features = detectX86Features();
    } else if (arch == CpuArchitecture::ARM64) {
        features = detectArmFeatures();
    }

    return features;
}

CpuVendor CpuCompatibility::detectCpuVendor() {
    // Simple detection based on CPUID or basic checks
#ifdef _MSC_VER
    int cpuInfo[4];
    __cpuid(cpuInfo, 0);
    char vendor[13];
    memcpy(vendor, &cpuInfo[1], 4);
    memcpy(vendor + 4, &cpuInfo[3], 4);
    memcpy(vendor + 8, &cpuInfo[2], 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "GenuineIntel") == 0) return CpuVendor::Intel;
    if (strcmp(vendor, "AuthenticAMD") == 0) return CpuVendor::AMD;
#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);

    char vendor[13];
    memcpy(vendor, &ebx, 4);
    memcpy(vendor + 4, &edx, 4);
    memcpy(vendor + 8, &ecx, 4);
    vendor[12] = '\0';

    if (strcmp(vendor, "GenuineIntel") == 0) return CpuVendor::Intel;
    if (strcmp(vendor, "AuthenticAMD") == 0) return CpuVendor::AMD;
#endif

#ifdef __APPLE__
    return CpuVendor::Apple;
#endif

    return CpuVendor::Unknown;
}

CpuArchitecture CpuCompatibility::detectCpuArchitecture() {
#if defined(_M_X64) || defined(__x86_64__)
    return CpuArchitecture::x86_64;
#elif defined(_M_ARM64) || defined(__aarch64__)
    return CpuArchitecture::ARM64;
#else
    return CpuArchitecture::Unknown;
#endif
}

bool CpuCompatibility::isLittleEndian() {
    uint32_t test = 1;
    return *(uint8_t*)&test == 1;
}

void* CpuCompatibility::alignedAlloc(size_t size, size_t alignment) {
#if defined(_MSC_VER) || defined(__MINGW32__)
    return _aligned_malloc(size, alignment);
#elif defined(__GNUC__) || defined(__clang__)
    return std::aligned_alloc(alignment, size);
#else
    // Fallback: allocate extra space and align manually
    void* ptr = malloc(size + alignment + sizeof(void*) - 1);
    if (!ptr) return nullptr;

    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t aligned_addr = (addr + sizeof(void*) + alignment - 1) & ~(alignment - 1);

    // Store original pointer for free
    *(reinterpret_cast<void**>(aligned_addr - sizeof(void*))) = ptr;

    return reinterpret_cast<void*>(aligned_addr);
#endif
}

void CpuCompatibility::alignedFree(void* ptr) {
    if (!ptr) return;

#if defined(_MSC_VER) || defined(__MINGW32__)
    _aligned_free(ptr);
#elif defined(__GNUC__) || defined(__clang__)
    std::free(ptr);
#else
    // Fallback: retrieve original pointer
    void* original = *(reinterpret_cast<void**>(reinterpret_cast<uintptr_t>(ptr) - sizeof(void*)));
    free(original);
#endif
}

std::string CpuCompatibility::getCpuInfoString() {
    CpuFeatures features = detectCpuFeatures();
    CpuVendor vendor = detectCpuVendor();
    CpuArchitecture arch = detectCpuArchitecture();

    std::string info = "CPU Info:\n";
    info += "Architecture: ";
    switch (arch) {
        case CpuArchitecture::x86_64: info += "x86_64\n"; break;
        case CpuArchitecture::ARM64: info += "ARM64\n"; break;
        default: info += "Unknown\n"; break;
    }

    info += "Vendor: ";
    switch (vendor) {
        case CpuVendor::Intel: info += "Intel\n"; break;
        case CpuVendor::AMD: info += "AMD\n"; break;
        case CpuVendor::ARM: info += "ARM\n"; break;
        case CpuVendor::Apple: info += "Apple\n"; break;
        default: info += "Unknown\n"; break;
    }

    info += "Cores: " + std::to_string(features.num_cores) + "\n";
    info += "Threads: " + std::to_string(features.num_threads) + "\n";

    if (arch == CpuArchitecture::x86_64) {
        info += "SIMD Support:\n";
        info += "  SSE: " + std::string(features.has_sse ? "Yes" : "No") + "\n";
        info += "  SSE2: " + std::string(features.has_sse2 ? "Yes" : "No") + "\n";
        info += "  SSE3: " + std::string(features.has_sse3 ? "Yes" : "No") + "\n";
        info += "  SSSE3: " + std::string(features.has_ssse3 ? "Yes" : "No") + "\n";
        info += "  SSE4.1: " + std::string(features.has_sse4_1 ? "Yes" : "No") + "\n";
        info += "  SSE4.2: " + std::string(features.has_sse4_2 ? "Yes" : "No") + "\n";
        info += "  AVX: " + std::string(features.has_avx ? "Yes" : "No") + "\n";
        info += "  AVX2: " + std::string(features.has_avx2 ? "Yes" : "No") + "\n";
        info += "  AVX-512: " + std::string(features.has_avx512 ? "Yes" : "No") + "\n";
        info += "  FMA: " + std::string(features.has_fma ? "Yes" : "No") + "\n";
    } else if (arch == CpuArchitecture::ARM64) {
        info += "NEON: " + std::string(features.has_neon ? "Yes" : "No") + "\n";
    }

    info += "Endianness: " + std::string(isLittleEndian() ? "Little" : "Big") + "\n";

    return info;
}

CpuFeatures CpuCompatibility::detectX86Features() {
    CpuFeatures features;

#ifdef _MSC_VER
    int cpuInfo[4];

    // Check SSE support
    __cpuid(cpuInfo, 1);
    features.has_sse = (cpuInfo[3] & (1 << 25)) != 0;
    features.has_sse2 = (cpuInfo[3] & (1 << 26)) != 0;
    features.has_sse3 = (cpuInfo[2] & (1 << 0)) != 0;
    features.has_ssse3 = (cpuInfo[2] & (1 << 9)) != 0;
    features.has_sse4_1 = (cpuInfo[2] & (1 << 19)) != 0;
    features.has_sse4_2 = (cpuInfo[2] & (1 << 20)) != 0;
    features.has_avx = (cpuInfo[2] & (1 << 28)) != 0;
    features.has_fma = (cpuInfo[2] & (1 << 12)) != 0;

    // Check AVX2 support
    __cpuidex(cpuInfo, 7, 0);
    features.has_avx2 = (cpuInfo[1] & (1 << 5)) != 0;

    // AVX-512 is more complex, simplified check
    features.has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;  // AVX-512 Foundation

#elif defined(__GNUC__) || defined(__clang__)
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    // Check basic features
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    features.has_sse = (edx & (1 << 25)) != 0;
    features.has_sse2 = (edx & (1 << 26)) != 0;
    features.has_sse3 = (ecx & (1 << 0)) != 0;
    features.has_ssse3 = (ecx & (1 << 9)) != 0;
    features.has_sse4_1 = (ecx & (1 << 19)) != 0;
    features.has_sse4_2 = (ecx & (1 << 20)) != 0;
    features.has_avx = (ecx & (1 << 28)) != 0;
    features.has_fma = (ecx & (1 << 12)) != 0;

    // Check extended features
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    features.has_avx2 = (ebx & (1 << 5)) != 0;
    features.has_avx512 = (ebx & (1 << 16)) != 0;  // AVX-512 Foundation

#endif

    return features;
}

CpuFeatures CpuCompatibility::detectArmFeatures() {
    CpuFeatures features;

    // Basic ARM64 detection - NEON is mandatory in ARM64
    features.has_neon = true;

    return features;
}

// SIMD-accelerated operations
float CpuCompatibility::vectorSum(const float* data, size_t size) {
    CpuFeatures features = detectCpuFeatures();

#ifdef __AVX__
    if (features.has_avx) {
        __m256 sum = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            sum = _mm256_add_ps(sum, vec);
        }

        // Horizontal sum
        float result[8];
        _mm256_storeu_ps(result, sum);
        float total = 0.0f;
        for (int j = 0; j < 8; ++j) total += result[j];

        // Add remaining elements
        for (; i < size; ++i) total += data[i];

        return total;
    }
#endif

#ifdef __ARM_NEON
    if (features.has_neon) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t vec = vld1q_f32(&data[i]);
            sum = vaddq_f32(sum, vec);
        }

        float result[4];
        vst1q_f32(result, sum);
        float total = result[0] + result[1] + result[2] + result[3];

        for (; i < size; ++i) total += data[i];

        return total;
    }
#endif

    // Fallback to standard sum
    return std::accumulate(data, data + size, 0.0f);
}

void CpuCompatibility::vectorAdd(float* result, const float* a, const float* b, size_t size) {
    CpuFeatures features = detectCpuFeatures();

#ifdef __AVX__
    if (features.has_avx) {
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vres = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result[i], vres);
        }

        for (; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
#endif

#ifdef __ARM_NEON
    if (features.has_neon) {
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t va = vld1q_f32(&a[i]);
            float32x4_t vb = vld1q_f32(&b[i]);
            float32x4_t vres = vaddq_f32(va, vb);
            vst1q_f32(&result[i], vres);
        }

        for (; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
#endif

    // Fallback
    for (size_t i = 0; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void CpuCompatibility::vectorScale(float* result, const float* data, float scale, size_t size) {
    CpuFeatures features = detectCpuFeatures();

#ifdef __AVX__
    if (features.has_avx) {
        __m256 vscale = _mm256_set1_ps(scale);
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            __m256 vres = _mm256_mul_ps(vec, vscale);
            _mm256_storeu_ps(&result[i], vres);
        }

        for (; i < size; ++i) {
            result[i] = data[i] * scale;
        }
        return;
    }
#endif

#ifdef __ARM_NEON
    if (features.has_neon) {
        float32x4_t vscale = vdupq_n_f32(scale);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            float32x4_t vec = vld1q_f32(&data[i]);
            float32x4_t vres = vmulq_f32(vec, vscale);
            vst1q_f32(&result[i], vres);
        }

        for (; i < size; ++i) {
            result[i] = data[i] * scale;
        }
        return;
    }
#endif

    // Fallback
    for (size_t i = 0; i < size; ++i) {
        result[i] = data[i] * scale;
    }
}

// Endianness handling
uint32_t CpuCompatibility::swapEndian32(uint32_t value) {
    return ((value >> 24) & 0xFF) |
           ((value >> 8) & 0xFF00) |
           ((value << 8) & 0xFF0000) |
           ((value << 24) & 0xFF000000);
}

uint64_t CpuCompatibility::swapEndian64(uint64_t value) {
    return ((value >> 56) & 0xFFULL) |
           ((value >> 40) & 0xFF00ULL) |
           ((value >> 24) & 0xFF0000ULL) |
           ((value >> 8) & 0xFF000000ULL) |
           ((value << 8) & 0xFF00000000ULL) |
           ((value << 24) & 0xFF0000000000ULL) |
           ((value << 40) & 0xFF000000000000ULL) |
           ((value << 56) & 0xFF00000000000000ULL);
}

void CpuCompatibility::swapEndian(float* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint32_t* intPtr = reinterpret_cast<uint32_t*>(&data[i]);
        *intPtr = swapEndian32(*intPtr);
    }
}

bool CpuCompatibility::shouldSwapEndianForFile() {
    // Assume little-endian files for now
    // In a real implementation, this would check file headers
    return !isLittleEndian();
}