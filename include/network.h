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
};

#endif // NETWORK_H
