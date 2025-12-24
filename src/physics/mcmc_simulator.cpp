#include "mcmc_simulator.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace NyxPhysics {

MCMCSimulator::MCMCSimulator() : rng_(std::random_device{}()), total_proposals_(0), accepted_proposals_(0) {
    // Default configuration
    config_.n_samples = 10000;
    config_.burn_in = 1000;
    config_.proposal_sigma = 1.0;
    config_.thinning = 10;
}

void MCMCSimulator::configure(const MCMCConfig& config) {
    config_ = config;
}

MCMCResults MCMCSimulator::run_metropolis_hastings(
    std::function<double(const std::vector<double>&)> log_target,
    const std::vector<double>& initial_state
) {
    reset_counters();

    std::vector<double> current = initial_state;
    double current_log_target = log_target(current);

    MCMCResults results;
    results.samples.reserve(config_.n_samples / config_.thinning);

    // Burn-in period
    for (int i = 0; i < config_.burn_in; ++i) {
        auto proposed = propose_move(current);
        double proposed_log_target = log_target(proposed);

        double log_ratio = proposed_log_target - current_log_target;
        if (metropolis_acceptance(log_ratio)) {
            current = proposed;
            current_log_target = proposed_log_target;
        }
    }

    // Main sampling
    for (int i = 0; i < config_.n_samples; ++i) {
        auto proposed = propose_move(current);
        double proposed_log_target = log_target(proposed);

        double log_ratio = proposed_log_target - current_log_target;
        if (metropolis_acceptance(log_ratio)) {
            current = proposed;
            current_log_target = proposed_log_target;
        }

        // Thinning and storage
        if (i % config_.thinning == 0) {
            results.samples.push_back(current);
            results.energies.push_back(-current_log_target); // Convert back from log
        }
    }

    results.acceptance_rate = calculate_acceptance_rate();
    results.diagnostics = generate_diagnostics(results);
    results.converged = check_convergence(results.samples);

    return results;
}

MCMCResults MCMCSimulator::simulate_ising_model(int lattice_size, double temperature, int n_steps) {
    // Simple 1D Ising model for demonstration
    // In reality, this would be more complex with 2D lattice

    auto log_target = [temperature](const std::vector<double>& spins) -> double {
        double energy = 0.0;
        // Simple nearest neighbor interaction
        for (size_t i = 0; i < spins.size() - 1; ++i) {
            energy -= spins[i] * spins[i + 1]; // Ferromagnetic coupling
        }
        // External field (simplified)
        for (double spin : spins) {
            energy -= 0.1 * spin; // Small field
        }
        return -energy / temperature; // Log probability proportional to -E/T
    };

    std::vector<double> initial_state(lattice_size, 1.0); // All spins up initially

    MCMCConfig config;
    config.n_samples = n_steps;
    config.burn_in = n_steps / 10;
    config.proposal_sigma = 0.1; // Small changes for spin flips
    config.thinning = 1;

    configure(config);

    return run_metropolis_hastings(log_target, initial_state);
}

MCMCResults MCMCSimulator::bayesian_parameter_estimation(
    const std::vector<double>& data,
    std::function<double(const std::vector<double>&, double)> likelihood,
    std::function<double(const std::vector<double>&)> prior,
    int n_steps
) {
    // Bayesian inference for a single parameter
    auto log_posterior = [&](const std::vector<double>& params) -> double {
        double log_prior = prior(params);
        double log_lik = 0.0;
        for (double datum : data) {
            log_lik += std::log(likelihood(params, datum));
        }
        return log_prior + log_lik;
    };

    std::vector<double> initial_state = {0.0}; // Single parameter

    MCMCConfig config;
    config.n_samples = n_steps;
    config.burn_in = n_steps / 10;
    config.thinning = 1;

    configure(config);

    return run_metropolis_hastings(log_posterior, initial_state);
}

std::vector<double> MCMCSimulator::propose_move(const std::vector<double>& current) {
    std::vector<double> proposed = current;
    std::normal_distribution<double> dist(0.0, config_.proposal_sigma);

    for (size_t i = 0; i < proposed.size(); ++i) {
        proposed[i] += dist(rng_);
    }

    return proposed;
}

double MCMCSimulator::metropolis_acceptance(double log_ratio) {
    total_proposals_++;

    if (log_ratio > 0) {
        accepted_proposals_++;
        return true;
    }

    double acceptance_prob = std::exp(log_ratio);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    if (uniform(rng_) < acceptance_prob) {
        accepted_proposals_++;
        return true;
    }

    return false;
}

void MCMCSimulator::reset_counters() {
    total_proposals_ = 0;
    accepted_proposals_ = 0;
}

double MCMCSimulator::calculate_acceptance_rate() const {
    if (total_proposals_ == 0) return 0.0;
    return static_cast<double>(accepted_proposals_) / total_proposals_;
}

std::string MCMCSimulator::generate_diagnostics(const MCMCResults& results) const {
    std::stringstream ss;

    ss << "MCMC Diagnostics:\n";
    ss << "Acceptance Rate: " << results.acceptance_rate * 100 << "%\n";
    ss << "Total Samples: " << results.samples.size() << "\n";
    ss << "Converged: " << (results.converged ? "Yes" : "No") << "\n";

    if (!results.samples.empty()) {
        // Calculate basic statistics
        std::vector<double> means(results.samples[0].size(), 0.0);
        for (const auto& sample : results.samples) {
            for (size_t i = 0; i < sample.size(); ++i) {
                means[i] += sample[i];
            }
        }
        for (double& mean : means) {
            mean /= results.samples.size();
        }

        ss << "Parameter Means: ";
        for (size_t i = 0; i < means.size(); ++i) {
            ss << means[i];
            if (i < means.size() - 1) ss << ", ";
        }
        ss << "\n";
    }

    return ss.str();
}

bool MCMCSimulator::check_convergence(const std::vector<std::vector<double>>& samples) const {
    if (samples.size() < 100) return false;

    // Simple convergence check: compare means of first and second halves
    size_t half = samples.size() / 2;
    std::vector<double> first_half_mean(samples[0].size(), 0.0);
    std::vector<double> second_half_mean(samples[0].size(), 0.0);

    for (size_t i = 0; i < half; ++i) {
        for (size_t j = 0; j < samples[i].size(); ++j) {
            first_half_mean[j] += samples[i][j];
        }
    }
    for (size_t i = half; i < samples.size(); ++i) {
        for (size_t j = 0; j < samples[i].size(); ++j) {
            second_half_mean[j] += samples[i][j];
        }
    }

    for (size_t j = 0; j < first_half_mean.size(); ++j) {
        first_half_mean[j] /= half;
        second_half_mean[j] /= (samples.size() - half);
        if (std::abs(first_half_mean[j] - second_half_mean[j]) > 0.1) { // Simple threshold
            return false;
        }
    }

    return true;
}

} // namespace NyxPhysics