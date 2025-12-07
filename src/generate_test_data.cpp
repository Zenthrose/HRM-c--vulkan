#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

// Configuration matching C++ AttentionConfig
const int hidden_size = 128;
const int head_dim = 64;
const int num_heads = 2;
const int num_key_value_heads = 2;
const bool causal = false;

// Example dimensions
const int batch_size = 1;
const int seq_len = 256;

// Simple random number generator for reproducibility
class SimpleRNG {
private:
    uint32_t state;
public:
    SimpleRNG(uint32_t seed = 42) : state(seed) {}

    float next_float() {
        // Simple LCG generator
        state = state * 1103515245 + 12345;
        return static_cast<float>(state) / static_cast<float>(UINT32_MAX);
    }

    float next_normal() {
        // Box-Muller transform for normal distribution
        float u1 = next_float();
        float u2 = next_float();
        return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265359f * u2);
    }
};

// Generate random tensor with normal distribution
std::vector<float> generate_random_tensor(int batch, int seq, int heads, int dim, SimpleRNG& rng) {
    std::vector<float> tensor(batch * seq * heads * dim);
    for (auto& val : tensor) {
        val = rng.next_normal();
    }
    return tensor;
}

// Simple attention computation (without FlashAttention)
std::vector<float> simple_attention(const std::vector<float>& q, const std::vector<float>& k,
                                   const std::vector<float>& v, bool causal) {
    int batch = batch_size;
    int seq = seq_len;
    int heads = num_heads;
    int kv_heads = num_key_value_heads;
    int dim = head_dim;

    float scale = std::sqrt(1.0f / dim);
    std::vector<float> output(batch * seq * heads * dim, 0.0f);

    // For each batch, head
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < heads; ++h) {
            // Compute attention scores
            std::vector<float> scores(seq * seq, 0.0f);

            for (int i = 0; i < seq; ++i) {
                for (int j = 0; j < seq; ++j) {
                    float score = 0.0f;
                    for (int d = 0; d < dim; ++d) {
                        int q_idx = b * seq * heads * dim + i * heads * dim + h * dim + d;
                        int k_idx = b * seq * kv_heads * dim + j * kv_heads * dim + (h % kv_heads) * dim + d;
                        score += q[q_idx] * k[k_idx];
                    }
                    scores[i * seq + j] = score * scale;

                    if (causal && j > i) {
                        scores[i * seq + j] = -INFINITY;
                    }
                }
            }

            // Softmax
            for (int i = 0; i < seq; ++i) {
                float max_score = *std::max_element(scores.begin() + i * seq, scores.begin() + (i + 1) * seq);
                float sum_exp = 0.0f;

                for (int j = 0; j < seq; ++j) {
                    scores[i * seq + j] = std::exp(scores[i * seq + j] - max_score);
                    sum_exp += scores[i * seq + j];
                }

                for (int j = 0; j < seq; ++j) {
                    scores[i * seq + j] /= sum_exp;
                }
            }

            // Apply to values
            for (int i = 0; i < seq; ++i) {
                for (int d = 0; d < dim; ++d) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq; ++j) {
                        int v_idx = b * seq * kv_heads * dim + j * kv_heads * dim + (h % kv_heads) * dim + d;
                        sum += scores[i * seq + j] * v[v_idx];
                    }
                    int out_idx = b * seq * heads * dim + i * heads * dim + h * dim + d;
                    output[out_idx] = sum;
                }
            }
        }
    }

    return output;
}

void save_tensor(const std::vector<float>& tensor, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    for (size_t i = 0; i < tensor.size(); ++i) {
        file << tensor[i];
        if (i < tensor.size() - 1) {
            file << "\n";
        }
    }

    std::cout << "Saved " << tensor.size() << " values to " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Generating Attention Test Data" << std::endl;
    std::cout << "==============================" << std::endl;

    // Parse output directory argument
    std::string output_dir = ".";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output-dir" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --output-dir <dir>   Output directory (default: .)" << std::endl;
            std::cout << "  --help               Show this help" << std::endl;
            return 0;
        }
    }

    // Create output directory if it doesn't exist
    fs::create_directories(output_dir);

    // Initialize RNG with same seed as Python version
    SimpleRNG rng(42);

    std::cout << "Generating test tensors..." << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Sequence length: " << seq_len << std::endl;
    std::cout << "Number of heads: " << num_heads << std::endl;
    std::cout << "Head dimension: " << head_dim << std::endl;
    std::cout << "KV heads: " << num_key_value_heads << std::endl;
    std::cout << "Causal: " << (causal ? "true" : "false") << std::endl;

    // Generate Q, K, V tensors
    auto query = generate_random_tensor(batch_size, seq_len, num_heads, head_dim, rng);
    auto key = generate_random_tensor(batch_size, seq_len, num_key_value_heads, head_dim, rng);
    auto value = generate_random_tensor(batch_size, seq_len, num_key_value_heads, head_dim, rng);

    std::cout << "Computing attention..." << std::endl;
    auto output = simple_attention(query, key, value, causal);

    // Save tensors
    std::string query_file = (fs::path(output_dir) / "test_data_query.txt").string();
    std::string key_file = (fs::path(output_dir) / "test_data_key.txt").string();
    std::string value_file = (fs::path(output_dir) / "test_data_value.txt").string();
    std::string output_file = (fs::path(output_dir) / "test_data_output.txt").string();

    save_tensor(query, query_file);
    save_tensor(key, key_file);
    save_tensor(value, value_file);
    save_tensor(output, output_file);

    std::cout << "Test data generation complete!" << std::endl;
    std::cout << "Files saved to: " << output_dir << std::endl;

    return 0;
}