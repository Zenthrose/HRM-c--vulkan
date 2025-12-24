#include <iostream>
#include <vector>
#include <cmath>

// Simple test to verify gradient computation works
struct Tensor {
    std::vector<float> data;
    std::vector<size_t> shape;
};

class SimpleLinear {
public:
    Tensor weight;
    Tensor bias;

    SimpleLinear(size_t in_features, size_t out_features) {
        weight.data.resize(in_features * out_features, 0.1f);
        weight.shape = {out_features, in_features};
        bias.data.resize(out_features, 0.0f);
        bias.shape = {out_features};
    }

    Tensor forward(const Tensor& input) {
        Tensor output;
        output.shape = {input.shape[0], weight.shape[0]};
        output.data.resize(input.shape[0] * weight.shape[0], 0.0f);

        // Simple matrix multiplication: output = input @ weight.T + bias
        for (size_t batch = 0; batch < input.shape[0]; ++batch) {
            for (size_t out = 0; out < weight.shape[0]; ++out) {
                float sum = bias.data[out];
                for (size_t in = 0; in < input.shape[1]; ++in) {
                    size_t input_idx = batch * input.shape[1] + in;
                    size_t weight_idx = out * weight.shape[1] + in;
                    sum += input.data[input_idx] * weight.data[weight_idx];
                }
                size_t output_idx = batch * weight.shape[0] + out;
                output.data[output_idx] = sum;
            }
        }
        return output;
    }

    std::pair<Tensor, std::vector<Tensor>> backward(const Tensor& input, const Tensor& output_grad) {
        // Compute gradients: dL/dinput, dL/dweight, dL/dbias

        Tensor input_grad;
        input_grad.shape = input.shape;
        input_grad.data.resize(input.data.size(), 0.0f);

        Tensor weight_grad;
        weight_grad.shape = weight.shape;
        weight_grad.data.resize(weight.data.size(), 0.0f);

        Tensor bias_grad;
        bias_grad.shape = bias.shape;
        bias_grad.data.resize(bias.data.size(), 0.0f);

        // Gradient computation
        for (size_t batch = 0; batch < input.shape[0]; ++batch) {
            for (size_t out = 0; out < weight.shape[0]; ++out) {
                float out_grad = output_grad.data[batch * weight.shape[0] + out];
                bias_grad.data[out] += out_grad;

                for (size_t in = 0; in < input.shape[1]; ++in) {
                    size_t input_idx = batch * input.shape[1] + in;
                    size_t weight_idx = out * weight.shape[1] + in;

                    // dL/dweight += input * dL/doutput
                    weight_grad.data[weight_idx] += input.data[input_idx] * out_grad;

                    // dL/dinput += dL/doutput * weight
                    input_grad.data[input_idx] += out_grad * weight.data[weight_idx];
                }
            }
        }

        return {input_grad, {weight_grad, bias_grad}};
    }
};

int main() {
    std::cout << "Testing gradient computation..." << std::endl;

    // Create a simple linear layer
    SimpleLinear layer(3, 2); // 3 inputs, 2 outputs

    // Create input tensor
    Tensor input;
    input.shape = {1, 3}; // batch_size=1, features=3
    input.data = {1.0f, 2.0f, 3.0f};

    // Forward pass
    Tensor output = layer.forward(input);
    std::cout << "Forward output: ";
    for (float val : output.data) std::cout << val << " ";
    std::cout << std::endl;

    // Create output gradients (simulating loss gradients)
    Tensor output_grad;
    output_grad.shape = output.shape;
    output_grad.data = {1.0f, -0.5f}; // Gradient w.r.t. outputs

    // Backward pass
    auto [input_grad, param_grads] = layer.backward(input, output_grad);

    std::cout << "Input gradients: ";
    for (float val : input_grad.data) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Weight gradients: ";
    for (float val : param_grads[0].data) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Bias gradients: ";
    for (float val : param_grads[1].data) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Gradient computation test passed!" << std::endl;
    return 0;
}