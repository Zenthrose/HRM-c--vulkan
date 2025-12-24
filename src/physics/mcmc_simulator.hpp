#pragma once

#include <vector>
#include <functional>
#include <random>
#include <string>
#include <memory>
#include <iostream>
#include <cmath>

namespace NyxPhysics {

// MCMC Configuration
struct MCMCConfig {
    int n_samples = 10000;
    int burn_in = 1000;
    double proposal_sigma = 1.0;
    int thinning = 10;
    std::vector<double> initial_state;
    bool use_vulkan = true;
};

// MCMC Results
struct MCMCResults {
    std::vector<std::vector<double>> samples;
    std::vector<double> energies;
    double acceptance_rate;
    std::string diagnostics;
    bool converged;
};

// MCMC Simulator Class
class MCMCSimulator {
public:
    MCMCSimulator();
    ~MCMCSimulator() = default;

    // Configure MCMC
    void configure(const MCMCConfig& config);

    // Run Metropolis-Hastings MCMC
    MCMCResults run_metropolis_hastings(
        std::function<double(const std::vector<double>&)> log_target,
        const std::vector<double>& initial_state
    );

    // Physics-specific simulations
    MCMCResults simulate_ising_model(int lattice_size, double temperature, int n_steps = 10000);
    MCMCResults bayesian_parameter_estimation(
        const std::vector<double>& data,
        std::function<double(const std::vector<double>&, double)> likelihood,
        std::function<double(const std::vector<double>&)> prior,
        int n_steps = 10000
    );

    // Utility functions
    double calculate_acceptance_rate() const;
    std::string generate_diagnostics(const MCMCResults& results) const;
    bool check_convergence(const std::vector<std::vector<double>>& samples) const;

private:
    MCMCConfig config_;
    std::mt19937 rng_;
    int total_proposals_;
    int accepted_proposals_;

    // Helper functions
    std::vector<double> propose_move(const std::vector<double>& current);
    double metropolis_acceptance(double log_ratio);
    void reset_counters();
};

} // namespace NyxPhysics