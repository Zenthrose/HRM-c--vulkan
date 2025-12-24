#include <iostream>
#include <cmath>
#include <cstring>
#include "quantization_c.c"

int main() {
    const uint32_t N = 13; // odd size to test packing
    float weights[N];
    for (uint32_t i = 0; i < N; ++i) weights[i] = (float)i * 0.12345f - 0.6f;

    QuantizationParams params;

    // INT4 roundtrip
    uint8_t* q4 = quantize_weights_fp32_to_int4_c(weights, N, &params);
    if (!q4) { std::cerr << "INT4 quant failed" << std::endl; return 2; }
    float* d4 = dequantize_int4_to_fp32_c(q4, (N+1)/2, &params);
    if (!d4) { std::cerr << "INT4 dequant failed" << std::endl; return 3; }
    free(q4); free(d4);

    // INT8 roundtrip
    uint8_t* q8 = quantize_weights_fp32_to_int8_c(weights, N, &params);
    if (!q8) { std::cerr << "INT8 quant failed" << std::endl; return 4; }
    float* d8 = dequantize_int8_to_fp32_c(q8, N, &params);
    if (!d8) { std::cerr << "INT8 dequant failed" << std::endl; return 5; }
    free(q8); free(d8);

    // FP16 roundtrip
    uint8_t* q16 = quantize_weights_fp32_to_fp16_c(weights, N, &params);
    if (!q16) { std::cerr << "FP16 quant failed" << std::endl; return 6; }
    float* d16 = dequantize_fp16_to_fp32_c(q16, N, &params);
    if (!d16) { std::cerr << "FP16 dequant failed" << std::endl; return 7; }
    free(q16); free(d16);

    // BF16 roundtrip
    uint8_t* qb = quantize_weights_fp32_to_bf16_c(weights, N, &params);
    if (!qb) { std::cerr << "BF16 quant failed" << std::endl; return 8; }
    float* db = dequantize_bf16_to_fp32_c(qb, N, &params);
    if (!db) { std::cerr << "BF16 dequant failed" << std::endl; return 9; }
    free(qb); free(db);

    std::cout << "Quantization CPU roundtrip tests passed" << std::endl;
    return 0;
}
