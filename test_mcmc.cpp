#include <iostream>
#include <vector>
#include <functional>
#include "src/physics/mcmc_simulator.hpp"

int main() {
    std::cout << "Testing MCMC Physics Simulator" << std::endl;

    NyxPhysics::MCMCSimulator simulator;

    // Configure for testing
    NyxPhysics::MCMCConfig config;
    config.n_samples = 1000;
    config.burn_in = 100;
    config.initial_state = {1.0, 0.0, 1.0};  // For Bayesian: m, b, sigma
    config.use_vulkan = false;  // Disable Vulkan for simple test
    simulator.configure(config);

    // Test Ising model
    std::cout << "Running Ising model simulation..." << std::endl;
    auto ising_results = simulator.simulate_ising_model(4, 2.0, 1000);

    std::cout << "Ising model results:" << std::endl;
    std::cout << "Samples: " << ising_results.samples.size() << std::endl;
    std::cout << "Acceptance rate: " << ising_results.acceptance_rate << std::endl;
    std::cout << "Converged: " << (ising_results.converged ? "Yes" : "No") << std::endl;

    // Test Bayesian parameter estimation
    std::cout << "Running Bayesian parameter estimation..." << std::endl;
    std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto bayesian_results = simulator.bayesian_parameter_estimation(
        data,
        [](const std::vector<double>& params, double x) {
            // Simple linear model: y = mx + b
            double m = params[0];
            double b = params[1];
            double sigma = params[2];
            double mean = m * x + b;
            return -0.5 * std::log(2 * M_PI * sigma * sigma) -
                   0.5 * (x - mean) * (x - mean) / (sigma * sigma);
        },
        [](const std::vector<double>& params) {
            // Simple prior: normal distributions
            double m = params[0], b = params[1], sigma = params[2];
            return -0.5 * (m * m + b * b) - std::log(sigma);  // log prior
        },
        1000  // n_steps
    );

    std::cout << "Bayesian estimation results:" << std::endl;
    std::cout << "Samples: " << bayesian_results.samples.size() << std::endl;
    std::cout << "Acceptance rate: " << bayesian_results.acceptance_rate << std::endl;
    std::cout << "Converged: " << (bayesian_results.converged ? "Yes" : "No") << std::endl;

    if (!ising_results.samples.empty()) {
        std::cout << "Sample energy: " << ising_results.energies[0] << std::endl;
    }

    std::cout << "MCMC tests completed successfully!" << std::endl;
    return 0;
}