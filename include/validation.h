#ifndef VALIDATION_H
#define VALIDATION_H

#include <cmath>
#include <stdexcept>
#include <string>

/**
 * Validation utilities for error handling and robustness.
 *
 * Provides functions to validate numerical values and configuration parameters
 * according to Requirements 12.1-12.5.
 */
namespace Validation {

/**
 * Validate that a value is finite (not NaN or Inf).
 *
 * Requirement 12.1: Detect NaN/Inf in mathematical operations
 * Requirement 12.5: Provide descriptive error messages with context
 *
 * @param value The value to check
 * @param context Description of where the value came from (operation, layer, etc.)
 * @throws std::runtime_error if value is NaN or Inf
 */
inline void validateFinite(double value, const std::string& context) {
    if (std::isnan(value)) {
        throw std::runtime_error(
            "NaN detected in " + context + ". "
            "This indicates numerical instability. "
            "Consider: reducing learning rate, checking input data, or using gradient clipping."
        );
    }
    if (std::isinf(value)) {
        throw std::runtime_error(
            "Inf detected in " + context + ". "
            "This indicates numerical overflow. "
            "Consider: reducing learning rate, normalizing inputs, or checking for exploding gradients."
        );
    }
}

/**
 * Validate that a configuration parameter is positive.
 *
 * Requirement 12.2: Validate configuration parameters are in valid range
 * Requirement 12.5: Provide descriptive error messages
 *
 * @param value The value to check
 * @param param_name Name of the parameter
 * @throws std::invalid_argument if value is not positive
 */
inline void validatePositive(double value, const std::string& param_name) {
    if (value <= 0.0) {
        throw std::invalid_argument(
            "Invalid configuration: " + param_name + " must be positive. "
            "Provided: " + std::to_string(value)
        );
    }
}

/**
 * Validate that a size parameter is positive.
 *
 * Requirement 12.2: Validate configuration parameters are in valid range
 * Requirement 12.5: Provide descriptive error messages
 *
 * @param value The value to check
 * @param param_name Name of the parameter
 * @throws std::invalid_argument if value is zero
 */
inline void validatePositiveSize(size_t value, const std::string& param_name) {
    if (value == 0) {
        throw std::invalid_argument(
            "Invalid configuration: " + param_name + " must be greater than zero. "
            "Provided: " + std::to_string(value)
        );
    }
}

} // namespace Validation

#endif // VALIDATION_H
