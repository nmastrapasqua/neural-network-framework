#include "layer.h"
#include <stdexcept>
#include <cmath>

/**
 * Construct a layer with specified dimensions and activation function.
 * Initializes weights to zero and biases to zero.
 * Weight initialization should be done separately using initialization methods.
 */
Layer::Layer(size_t input_size, size_t output_size,
             std::shared_ptr<ActivationFunction> activation)
    : weights_(output_size, input_size, 0.0),
      biases_(output_size, 0.0),
      activation_(activation),
      last_input_(input_size, 0.0),
      last_weighted_sum_(output_size, 0.0),
      last_output_(output_size, 0.0)
{
    if (input_size == 0) {
        throw std::invalid_argument("Layer input_size must be greater than zero");
    }
    if (output_size == 0) {
        throw std::invalid_argument("Layer output_size must be greater than zero");
    }
    if (!activation) {
        throw std::invalid_argument("Layer activation function cannot be null");
    }
}

/**
 * Perform forward propagation through this layer.
 *
 * Algorithm:
 * 1. Validate input dimensions
 * 2. Compute weighted sum: z = W*input + b
 * 3. Apply activation function element-wise: output = activation(z)
 * 4. Cache input, weighted sum, and output for backpropagation
 *
 * @param input Input vector (size must match input_size)
 * @return Output vector after applying weights, biases, and activation
 */
Vector Layer::forward(const Vector& input) {
    // Validate input dimensions
    if (input.size() != weights_.cols()) {
        throw std::invalid_argument(
            "Layer forward: input size " + std::to_string(input.size()) +
            " does not match expected size " + std::to_string(weights_.cols())
        );
    }

    // Cache input for backpropagation
    last_input_ = input;

    // Compute weighted sum: z = W*input + b
    last_weighted_sum_ = weights_ * input;

    // Add biases
    for (size_t i = 0; i < last_weighted_sum_.size(); ++i) {
        last_weighted_sum_[i] += biases_[i];
    }

    // Apply activation function element-wise
    last_output_ = Vector(last_weighted_sum_.size());
    for (size_t i = 0; i < last_weighted_sum_.size(); ++i) {
        last_output_[i] = activation_->activate(last_weighted_sum_[i]);
    }

    return last_output_;
}

/**
 * Get the last input that was passed to forward().
 * Used in backpropagation to compute gradients.
 */
const Vector& Layer::getLastInput() const {
    return last_input_;
}

/**
 * Get the last output produced by forward().
 * This is the activated output: activation(z).
 */
const Vector& Layer::getLastOutput() const {
    return last_output_;
}

/**
 * Get the last activation (same as getLastOutput).
 * Provided for API consistency with design document.
 */
const Vector& Layer::getLastActivation() const {
    return last_output_;
}

/**
 * Get const reference to weights matrix.
 */
const Matrix& Layer::getWeights() const {
    return weights_;
}

/**
 * Get mutable reference to weights matrix.
 * Used for weight initialization and updates during training.
 */
Matrix& Layer::getWeights() {
    return weights_;
}

/**
 * Get const reference to biases vector.
 */
const Vector& Layer::getBiases() const {
    return biases_;
}

/**
 * Get mutable reference to biases vector.
 * Used for bias initialization and updates during training.
 */
Vector& Layer::getBiases() {
    return biases_;
}

/**
 * Get the input size of this layer.
 */
size_t Layer::inputSize() const {
    return weights_.cols();
}

/**
 * Get the output size of this layer.
 */
size_t Layer::outputSize() const {
    return weights_.rows();
}

/**
 * Get the name of the activation function used by this layer.
 */
std::string Layer::activationName() const {
    return activation_->name();
}

/**
 * Initialize weights with uniform random values in range [min, max].
 * Uses the Matrix::randomize() method for uniform distribution.
 *
 * @param min Minimum value for weight initialization
 * @param max Maximum value for weight initialization
 */
void Layer::initializeWeights(double min, double max) {
    if (min >= max) {
        throw std::invalid_argument(
            "Layer initializeWeights: min (" + std::to_string(min) +
            ") must be less than max (" + std::to_string(max) + ")"
        );
    }
    weights_.randomize(min, max);
}

/**
 * Initialize weights using Xavier initialization.
 *
 * Xavier initialization is designed for sigmoid and tanh activation functions.
 * It helps maintain the variance of activations across layers.
 *
 * Formula: weights ~ Uniform[-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
 *
 * Reference: Glorot & Bengio (2010), "Understanding the difficulty of training
 * deep feedforward neural networks"
 *
 * @param fan_in Number of input units (input_size)
 * @param fan_out Number of output units (output_size)
 */
void Layer::initializeXavier(size_t fan_in, size_t fan_out) {
    if (fan_in == 0 || fan_out == 0) {
        throw std::invalid_argument(
            "Layer initializeXavier: fan_in and fan_out must be greater than zero"
        );
    }

    double limit = std::sqrt(6.0 / (fan_in + fan_out));
    weights_.randomize(-limit, limit);
}

/**
 * Initialize weights using He initialization.
 *
 * He initialization is designed for ReLU activation functions.
 * It accounts for the fact that ReLU zeros out half of the activations.
 *
 * Formula: weights ~ Uniform[-sqrt(2/fan_in), sqrt(2/fan_in)]
 *
 * Reference: He et al. (2015), "Delving Deep into Rectifiers: Surpassing
 * Human-Level Performance on ImageNet Classification"
 *
 * @param fan_in Number of input units (input_size)
 */
void Layer::initializeHe(size_t fan_in) {
    if (fan_in == 0) {
        throw std::invalid_argument(
            "Layer initializeHe: fan_in must be greater than zero"
        );
    }

    double limit = std::sqrt(2.0 / fan_in);
    weights_.randomize(-limit, limit);
}
