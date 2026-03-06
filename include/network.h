#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <memory>
#include <cstddef>
#include "layer.h"
#include "vector.h"
#include "activation.h"

// Forward declaration
class LossFunction;

/**
 * Network class represents a feedforward neural network.
 *
 * A network consists of:
 * - Multiple layers connected in sequence
 * - Each layer has weights, biases, and an activation function
 *
 * The network provides:
 * - Architecture building via addLayer()
 * - Forward propagation via predict()
 * - Topology inspection via numLayers() and getTopology()
 *
 * Requirements validated:
 * - 1.1: Network creation with arbitrary number of layers
 * - 1.2: Validate neuron count > 0 when adding layers
 * - 1.3: Store complete topology (input, hidden, output dimensions)
 * - 1.4: Automatically connect layers in sequence
 * - 1.5: Provide methods to query network structure
 * - 4.1: Execute forward pass through all layers
 * - 4.5: Return output of final layer
 * - 4.6: Validate input dimensions match network input layer
 */
class Network {
public:
    /**
     * Construct an empty network.
     * Layers must be added using addLayer().
     */
    Network();

    /**
     * Add a layer to the network.
     *
     * Validates:
     * - input_size and output_size must be > 0 (Requirement 1.2)
     * - If not the first layer, input_size must match previous layer's output_size (Requirement 1.4)
     *
     * @param input_size Number of inputs to this layer
     * @param output_size Number of outputs (neurons) in this layer
     * @param activation Shared pointer to activation function
     * @throws std::invalid_argument if validation fails
     */
    void addLayer(size_t input_size, size_t output_size,
                  std::shared_ptr<ActivationFunction> activation);

    /**
     * Perform forward propagation through the entire network.
     *
     * Validates:
     * - Network must have at least one layer
     * - Input size must match first layer's input size (Requirement 4.6)
     *
     * Algorithm:
     * 1. Pass input through first layer
     * 2. Pass each layer's output to the next layer
     * 3. Return final layer's output (Requirement 4.5)
     *
     * @param input Input vector (size must match network input layer)
     * @return Output vector from final layer
     * @throws std::invalid_argument if validation fails
     */
    Vector predict(const Vector& input);

    /**
     * Get the number of layers in the network.
     *
     * @return Number of layers (Requirement 1.5)
     */
    size_t numLayers() const;

    /**
     * Get the complete network topology.
     *
     * Returns a vector containing the size of each layer:
     * [input_size_of_layer_0, output_size_of_layer_0, output_size_of_layer_1, ...]
     *
     * For a network with layers [2->4, 4->3, 3->1], returns [2, 4, 3, 1]
     *
     * @return Vector of layer sizes representing the complete topology (Requirement 1.3, 1.5)
     */
    std::vector<size_t> getTopology() const;

    /**
     * Train the network on a dataset using backpropagation with batch support.
     *
     * Implements training loop with configurable batch size:
     * - batch_size = 1: Stochastic Gradient Descent (SGD) - update after each example
     * - batch_size = 32 (typical): Mini-batch Gradient Descent - update after 32 examples
     * - batch_size = dataset_size: Batch Gradient Descent - update once per epoch
     *
     * Algorithm:
     * For each epoch:
     *   For each batch in dataset:
     *     Initialize gradient accumulator
     *     For each example in batch:
     *       1. Forward pass: compute output
     *       2. Compute loss
     *       3. Backpropagation: compute gradients
     *       4. Accumulate gradients
     *     Average accumulated gradients over batch size
     *     Update parameters using averaged gradients
     *   Compute and record average loss for epoch
     *
     * Requirements validated:
     * - 8.1: Accept training dataset with input-target pairs
     * - 8.2: Iterate through dataset for specified number of epochs
     * - 8.3: Support configurable batch_size parameter
     * - 8.4: Execute SGD when batch_size = 1
     * - 8.5: Accumulate and average gradients when batch_size > 1
     * - 8.6: Execute batch gradient descent when batch_size = dataset_size
     * - 8.7: Calculate and store average loss per epoch
     * - 8.10: Validate batch_size is > 0 and <= dataset size
     *
     * @param inputs Vector of input vectors (training examples)
     * @param targets Vector of target output vectors (labels)
     * @param epochs Number of complete passes through the dataset
     * @param learning_rate Learning rate (η) for gradient descent
     * @param loss_function Loss function to measure prediction error
     * @param batch_size Number of examples per batch (default = 1 for SGD)
     * @return Vector of average loss values (one per epoch)
     * @throws std::invalid_argument if validation fails
     */
    std::vector<double> train(const std::vector<Vector>& inputs,
                              const std::vector<Vector>& targets,
                              size_t epochs,
                              double learning_rate,
                              LossFunction& loss_function,
                              size_t batch_size = 1);

    /**
     * Get a reference to a specific layer for weight initialization or inspection.
     *
     * @param index Layer index (0-based)
     * @return Reference to the layer
     * @throws std::out_of_range if index is invalid
     */
    Layer& getLayer(size_t index);

    /**
     * Get a const reference to a specific layer for inspection.
     *
     * @param index Layer index (0-based)
     * @return Const reference to the layer
     * @throws std::out_of_range if index is invalid
     */
    const Layer& getLayer(size_t index) const;

private:
    std::vector<Layer> layers_;

    /**
     * Perform backpropagation to compute gradients for all weights and biases.
     *
     * Implements the backpropagation algorithm:
     * 1. Compute output layer delta: δ^L = loss.gradient() ⊙ activation.derivative(z^L)
     * 2. Propagate delta backward: δ^l = (W^(l+1))^T * δ^(l+1) ⊙ activation.derivative(z^l)
     * 3. Compute weight gradients: ∂L/∂W^l = δ^l * (a^(l-1))^T
     * 4. Compute bias gradients: ∂L/∂b^l = δ^l
     *
     * Uses cached values from forward pass (last_input_, last_weighted_sum_, last_output_)
     *
     * Requirements validated:
     * - 6.1: Calculate gradient of loss function w.r.t. all weights
     * - 6.2: Calculate gradient of loss function w.r.t. all biases
     * - 6.3: Propagate gradients backward through all layers using chain rule
     * - 6.4: Use intermediate outputs memorized during forward pass
     * - 6.5: Apply activation function derivative during backpropagation
     * - 6.6: Store all computed gradients for parameter update
     *
     * @param target Target output vector
     * @param loss_function Loss function to compute initial gradient
     * @param weight_gradients Output vector to store weight gradients (one matrix per layer)
     * @param bias_gradients Output vector to store bias gradients (one vector per layer)
     */
    void backpropagate(const Vector& target,
                       LossFunction& loss_function,
                       std::vector<Matrix>& weight_gradients,
                       std::vector<Vector>& bias_gradients);

    /**
     * Update network parameters using computed gradients.
     *
     * Applies gradient descent update rule:
     * θ_new = θ_old - η * gradient
     *
     * where θ represents weights or biases, η is the learning rate.
     *
     * Requirements validated:
     * - 7.1: Update all weights using gradient descent
     * - 7.2: Update all biases using gradient descent
     * - 7.3: Apply learning rate to parameter updates
     * - 7.5: Execute update according to formula: param_new = param_old - learning_rate * gradient
     *
     * @param weight_gradients Gradients for weights (one matrix per layer)
     * @param bias_gradients Gradients for biases (one vector per layer)
     * @param learning_rate Learning rate (η) to scale gradient updates
     */
    void updateParameters(const std::vector<Matrix>& weight_gradients,
                          const std::vector<Vector>& bias_gradients,
                          double learning_rate);
};

#endif // NETWORK_H
