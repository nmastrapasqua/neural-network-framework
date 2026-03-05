#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <cstddef>
#include "vector.h"
#include "matrix.h"
#include "activation.h"

/**
 * Layer class represents a single layer in a neural network.
 *
 * A layer consists of:
 * - Weights matrix (output_size x input_size)
 * - Biases vector (output_size)
 * - Activation function
 *
 * The layer performs forward propagation: z = W*input + b, then a = activation(z)
 * It caches intermediate values (input, weighted sum, output) for backpropagation.
 */
class Layer {
public:
    /**
     * Construct a layer with specified dimensions and activation function.
     *
     * @param input_size Number of inputs to this layer
     * @param output_size Number of outputs (neurons) in this layer
     * @param activation Shared pointer to activation function
     */
    Layer(size_t input_size, size_t output_size,
          std::shared_ptr<ActivationFunction> activation);

    /**
     * Perform forward propagation through this layer.
     * Computes: z = W*input + b, then output = activation(z)
     * Caches input, weighted sum, and output for backpropagation.
     *
     * @param input Input vector (size must match input_size)
     * @return Output vector after applying weights, biases, and activation
     */
    Vector forward(const Vector& input);

    // Accessors for cached values (used in backpropagation)
    const Vector& getLastInput() const;
    const Vector& getLastOutput() const;
    const Vector& getLastActivation() const;
    const Vector& getLastWeightedSum() const;  // z = W*input + b (before activation)

    // Accessors for parameters
    const Matrix& getWeights() const;
    Matrix& getWeights();
    const Vector& getBiases() const;
    Vector& getBiases();

    // Weight initialization methods
    /**
     * Initialize weights with uniform random values in range [min, max].
     *
     * @param min Minimum value for weight initialization
     * @param max Maximum value for weight initialization
     */
    void initializeWeights(double min, double max);

    /**
     * Initialize weights using Xavier initialization.
     * Suitable for sigmoid and tanh activation functions.
     * Range: [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
     *
     * @param fan_in Number of input units (input_size)
     * @param fan_out Number of output units (output_size)
     */
    void initializeXavier(size_t fan_in, size_t fan_out);

    /**
     * Initialize weights using He initialization.
     * Suitable for ReLU activation function.
     * Range: [-sqrt(2/fan_in), sqrt(2/fan_in)]
     *
     * @param fan_in Number of input units (input_size)
     */
    void initializeHe(size_t fan_in);

    // Layer properties
    size_t inputSize() const;
    size_t outputSize() const;
    std::string activationName() const;
    std::shared_ptr<ActivationFunction> getActivation() const;

private:
    Matrix weights_;  // output_size x input_size
    Vector biases_;   // output_size
    std::shared_ptr<ActivationFunction> activation_;

    // Cached values for backpropagation
    Vector last_input_;
    Vector last_weighted_sum_;
    Vector last_output_;
};

#endif // LAYER_H
