#include <gtest/gtest.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include "core/attention.hpp"

using namespace std;

static vector<float> read_float_vector(const string& path) {
    vector<float> out;
    ifstream f(path);
    float v;
    while (f >> v) out.push_back(v);
    return out;
}

// CPU forward matching implementation used in backward
static vector<float> cpu_attention_forward(const vector<float>& q_flat, const vector<float>& k_flat, const vector<float>& v_flat, const AttentionConfig& cfg) {
    const uint32_t B = cfg.batch_size;
    const uint32_t S = cfg.seq_len;
    const uint32_t H = cfg.num_heads;
    const uint32_t Hk = cfg.num_key_value_heads;
    const uint32_t D = cfg.head_dim;

    auto idx4 = [](uint32_t b, uint32_t s, uint32_t h, uint32_t d, uint32_t B, uint32_t S, uint32_t H, uint32_t D) {
        return (((size_t)b * S + s) * H + h) * D + d;
    };

    vector<float> out(static_cast<size_t>(B) * S * H * D, 0.0f);
    const float scale = 1.0f / sqrtf((float)D);

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t hk = h % Hk;
            // scores
            vector<float> scores(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float s_val = 0.0f;
                    for (uint32_t d = 0; d < D; ++d) {
                        size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                        size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        s_val += q_flat[q_index] * k_flat[k_index];
                    }
                    scores[qi * S + ki] = s_val * scale;
                }
            }
            // softmax and output
            for (uint32_t qi = 0; qi < S; ++qi) {
                float m = -numeric_limits<float>::infinity();
                for (uint32_t ki = 0; ki < S; ++ki) m = max(m, scores[qi * S + ki]);
                float sum = 0.0f;
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float e = expf(scores[qi * S + ki] - m);
                    scores[qi * S + ki] = e;
                    sum += e;
                }
                for (uint32_t ki = 0; ki < S; ++ki) scores[qi * S + ki] /= sum;

                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t ki = 0; ki < S; ++ki) {
                        size_t v_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        acc += scores[qi * S + ki] * v_flat[v_index];
                    }
                    size_t out_index = idx4(b, qi, h, d, B, S, H, D);
                    out[out_index] = acc;
                }
            }
        }
    }
    return out;
}

TEST(AttentionCPUGrad, FiniteDiffQ) {
    // Paths relative to test binary working dir where CMake copies data
    auto q = read_float_vector("tests/data/test_data_query.txt");
    auto k = read_float_vector("tests/data/test_data_key.txt");
    auto v = read_float_vector("tests/data/test_data_value.txt");

    AttentionConfig cfg;
    cfg.batch_size = 1;
    cfg.seq_len = 256;
    cfg.head_dim = 64;
    cfg.num_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.causal = false;

    // Build hidden_states as concatenation
    vector<float> hidden;
    hidden.insert(hidden.end(), q.begin(), q.end());
    hidden.insert(hidden.end(), k.begin(), k.end());
    hidden.insert(hidden.end(), v.begin(), v.end());

    Tensor hidden_t; hidden_t.data = hidden; hidden_t.shape = {cfg.batch_size, cfg.seq_len, cfg.num_heads * cfg.head_dim};

    // Use ones as output gradient
    size_t out_elems = (size_t)cfg.batch_size * cfg.seq_len * cfg.num_heads * cfg.head_dim;
    Tensor outg; outg.data.assign(out_elems, 1.0f); outg.shape = {cfg.batch_size, cfg.seq_len, cfg.num_heads * cfg.head_dim};

    auto [inp_grad, pgrads] = cpu_attention_backward(hidden_t, outg, cfg);

    // finite diff for first few elements in Q
    double eps = 1e-3;
    int checks = 5;
    for (int i = 0; i < checks; ++i) {
        vector<float> hidden_p = hidden;
        vector<float> hidden_m = hidden;
        hidden_p[i] += eps;
        hidden_m[i] -= eps;
        // split q/k/v
        vector<float> q_p(q.begin(), q.end());
        q_p[i] += eps;
        vector<float> q_m(q.begin(), q.end());
        q_m[i] -= eps;
        auto out_p = cpu_attention_forward(q_p, k, v, cfg);
        auto out_m = cpu_attention_forward(q_m, k, v, cfg);
        // loss = sum(out * outg) where outg = 1 -> sum(out)
        double loss_p = accumulate(out_p.begin(), out_p.end(), 0.0);
        double loss_m = accumulate(out_m.begin(), out_m.end(), 0.0);
        double num_grad = (loss_p - loss_m) / (2 * eps);
        double anal_grad = inp_grad.data[i];
        double diff = fabs(num_grad - anal_grad);
        double denom = max(1.0, fabs(num_grad) + fabs(anal_grad));
        double rel = diff / denom;
        // allow some tolerance
        EXPECT_LE(rel, 1e-2) << "Index " << i << " num=" << num_grad << " anal=" << anal_grad;
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
