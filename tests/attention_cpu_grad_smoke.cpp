// Standalone smoke test to validate CPU attention backward numerically for Q,K,V
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <random>
#include <cassert>

#include "core/attention.hpp"

using namespace std;

static vector<float> read_float_file(const string& path) {
    ifstream in(path);
    if (!in) throw runtime_error("failed to open " + path);
    vector<float> out;
    float v;
    while (in >> v) out.push_back(v);
    return out;
}

// CPU forward matching generate_attention_test_data.simple_attention
static vector<float> cpu_attention_forward(const Tensor& hidden_states, const AttentionConfig& cfg) {
    const uint32_t B = cfg.batch_size;
    const uint32_t S = cfg.seq_len;
    const uint32_t H = cfg.num_heads;
    const uint32_t Hk = cfg.num_key_value_heads;
    const uint32_t D = cfg.head_dim;

    size_t q_elems = static_cast<size_t>(B) * S * H * D;
    size_t k_elems = static_cast<size_t>(B) * S * Hk * D;
    size_t v_elems = k_elems;
    const float* q_ptr = hidden_states.data.data();
    const float* k_ptr = q_ptr + q_elems;
    const float* v_ptr = k_ptr + k_elems;

    vector<float> out(static_cast<size_t>(B) * S * H * D, 0.0f);
    const float scale = 1.0f / sqrt(static_cast<float>(D));

    auto idx4 = [](uint32_t b, uint32_t s, uint32_t h, uint32_t d, uint32_t B, uint32_t S, uint32_t H, uint32_t D){
        return (((size_t)b * S + s) * H + h) * D + d;
    };

    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            uint32_t hk = h % Hk;
            // Compute scores and softmax
            vector<float> scores(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t ki = 0; ki < S; ++ki) {
                    float s = 0.0f;
                    for (uint32_t d = 0; d < D; ++d) {
                        size_t q_index = idx4(b, qi, h, d, B, S, H, D);
                        size_t k_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        s += q_ptr[q_index] * k_ptr[k_index];
                    }
                    scores[qi * S + ki] = s * scale;
                }
            }

            vector<float> attn(S * S);
            for (uint32_t qi = 0; qi < S; ++qi) {
                float m = -numeric_limits<float>::infinity();
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) continue;
                    m = max(m, scores[qi * S + ki]);
                }
                float sum = 0.0f;
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) { attn[qi * S + ki] = 0.0f; continue; }
                    float e = exp(scores[qi * S + ki] - m);
                    attn[qi * S + ki] = e;
                    sum += e;
                }
                for (uint32_t ki = 0; ki < S; ++ki) {
                    if (cfg.causal && ki > qi) continue;
                    attn[qi * S + ki] /= sum;
                }
            }

            // output = attn * V
            for (uint32_t qi = 0; qi < S; ++qi) {
                for (uint32_t d = 0; d < D; ++d) {
                    float acc = 0.0f;
                    for (uint32_t ki = 0; ki < S; ++ki) {
                        size_t v_index = idx4(b, ki, hk, d, B, S, Hk, D);
                        acc += attn[qi * S + ki] * v_ptr[v_index];
                    }
                    size_t out_index = idx4(b, qi, h, d, B, S, H, D);
                    out[out_index] = acc;
                }
            }
        }
    }
    return out;
}

int main() {
    // Paths relative to project root (CMake copies files into test dir in many setups)
    const string qpath = "tests/data/test_data_query.txt";
    const string kpath = "tests/data/test_data_key.txt";
    const string vpath = "tests/data/test_data_value.txt";

    auto q = read_float_file(qpath);
    auto k = read_float_file(kpath);
    auto v = read_float_file(vpath);

    // Basic config matching generate_attention_test_data
    AttentionConfig cfg{};
    cfg.batch_size = 1;
    cfg.seq_len = 256;
    cfg.head_dim = 64;
    cfg.num_heads = 2;
    cfg.num_key_value_heads = 2;
    cfg.causal = false;

    // Build hidden_states: Q|K|V
    Tensor hidden;
    hidden.data.reserve(q.size() + k.size() + v.size());
    hidden.data.insert(hidden.data.end(), q.begin(), q.end());
    hidden.data.insert(hidden.data.end(), k.begin(), k.end());
    hidden.data.insert(hidden.data.end(), v.begin(), v.end());
    hidden.shape = {cfg.batch_size, cfg.seq_len, cfg.num_heads * cfg.head_dim};

    // Random output grad
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Tensor outg;
    outg.shape = {cfg.batch_size, cfg.seq_len, cfg.num_heads * cfg.head_dim};
    outg.data.resize(static_cast<size_t>(cfg.batch_size) * cfg.seq_len * cfg.num_heads * cfg.head_dim);
    for (auto &x : outg.data) x = dist(rng);

    // Analytical backward
    auto analytical = cpu_attention_backward(hidden, outg, cfg).first; // concatenated dq|dk|dv

    // Choose indices to check across Q,K,V (first, middle, last, and a few offsets)
    size_t total = hidden.data.size();
    vector<size_t> checks;
    checks.push_back(0);
    checks.push_back(1);
    checks.push_back(total/4);
    checks.push_back(total/2);
    checks.push_back((3*total)/4);
    checks.push_back(total-1);

    const double eps = 1e-2; // use slightly larger eps for numerical stability
    const double ABS_TOL = 2e-3;
    const double REL_TOL = 5e-3;
    bool ok = true;

    // Finite-difference: loss = sum(out * outg)
    auto loss_for = [&](const Tensor& h) {
        auto out = cpu_attention_forward(h, cfg);
        double s = 0.0;
        size_t n = min(out.size(), outg.data.size());
        for (size_t i = 0; i < n; ++i) s += static_cast<double>(out[i]) * static_cast<double>(outg.data[i]);
        return s;
    };

    // Add a deterministic set of extra indices for better coverage
    std::vector<size_t> extra;
    std::mt19937 idx_rng(123);
    std::uniform_int_distribution<size_t> idx_dist(0, total - 1);
    for (int i = 0; i < 10; ++i) extra.push_back(idx_dist(idx_rng));
    for (size_t idx : checks) extra.push_back(idx);

    for (size_t idx : extra) {
        Tensor h_plus = hidden;
        Tensor h_minus = hidden;
        h_plus.data[idx] += eps;
        h_minus.data[idx] -= eps;
        double Lp = loss_for(h_plus);
        double Lm = loss_for(h_minus);
        double numgrad = (Lp - Lm) / (2.0 * eps);
        double angrad = analytical.data[idx];
        double abs_err = fabs(numgrad - angrad);
        double rel_err = abs_err / (fabs(numgrad) + fabs(angrad) + 1e-12);
        cout << "idx=" << idx << " num=" << numgrad << " an=" << angrad << " abs_err=" << abs_err << " rel_err=" << rel_err << "\n";
        if (!(abs_err < ABS_TOL || rel_err < REL_TOL)) {
            cerr << "Gradient check failed at idx=" << idx << "\n";
            ok = false;
        }
    }

    if (!ok) {
        cerr << "attention_cpu_grad_smoke: FAILED" << endl;
        return 1;
    }
    cout << "attention_cpu_grad_smoke: PASSED" << endl;
    return 0;
}
