#include "loss.h"
#include <stdexcept>
#include <cmath>

// ============================================================================
// MeanSquaredError Implementation
// ============================================================================

double MeanSquaredError::compute(const Vector& predicted, const Vector& target) const {
    // Validate dimensions
    if (predicted.size() != target.size()) {
        throw std::invalid_argument(
            "MSE compute failed: predicted size (" + std::to_string(predicted.size()) +
            ") != target size (" + std::to_string(target.size()) + ")"
        );
    }

    // Compute mean squared error: (1/n) * sum((predicted - target)^2)
    double sum_squared_error = 0.0;
    size_t n = predicted.size();

    for (size_t i = 0; i < n; ++i) {
        double diff = predicted[i] - target[i];
        sum_squared_error += diff * diff;
    }

    return sum_squared_error / static_cast<double>(n);
}

Vector MeanSquaredError::gradient(const Vector& predicted, const Vector& target) const {
    // Validate dimensions
    if (predicted.size() != target.size()) {
        throw std::invalid_argument(
            "MSE gradient failed: predicted size (" + std::to_string(predicted.size()) +
            ") != target size (" + std::to_string(target.size()) + ")"
        );
    }

    // Compute gradient: (2/n) * (predicted - target)
    size_t n = predicted.size();
    Vector grad(n);
    double scale = 2.0 / static_cast<double>(n);

    for (size_t i = 0; i < n; ++i) {
        grad[i] = scale * (predicted[i] - target[i]);
    }

    return grad;
}

// ============================================================================
// CrossEntropy Implementation
// ============================================================================

double CrossEntropy::compute(const Vector& predicted, const Vector& target) const {
    // Validate dimensions
    if (predicted.size() != target.size()) {
        throw std::invalid_argument(
            "CrossEntropy compute failed: predicted size (" + std::to_string(predicted.size()) +
            ") != target size (" + std::to_string(target.size()) + ")"
        );
    }

    // Compute cross-entropy: -sum(target * log(predicted))
    // Add epsilon to predicted values to avoid log(0)
    double loss = 0.0;
    size_t n = predicted.size();

    for (size_t i = 0; i < n; ++i) {
        // Clamp predicted value to [EPSILON, 1.0] for numerical stability
        double pred_clamped = std::max(EPSILON, std::min(1.0, predicted[i]));
        loss -= target[i] * std::log(pred_clamped);
    }

    return loss;
}

Vector CrossEntropy::gradient(const Vector& predicted, const Vector& target) const {
    // Validate dimensions
    if (predicted.size() != target.size()) {
        throw std::invalid_argument(
            "CrossEntropy gradient failed: predicted size (" + std::to_string(predicted.size()) +
            ") != target size (" + std::to_string(target.size()) + ")"
        );
    }

    // Compute gradient: -target / predicted
    // Add epsilon to avoid division by zero
    size_t n = predicted.size();
    Vector grad(n);

    for (size_t i = 0; i < n; ++i) {
        // Clamp predicted value to avoid division by zero
        double pred_clamped = std::max(EPSILON, predicted[i]);
        grad[i] = -target[i] / pred_clamped;
    }

    return grad;
}
