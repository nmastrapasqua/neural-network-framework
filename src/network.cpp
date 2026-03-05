#include "network.h"
#include "loss.h"
#include <stdexcept>
#include <algorithm>

/**
 * Construct an empty network.
 * Layers must be added using addLayer().
 */
Network::Network() : layers_() {
    // Empty network, layers will be added via addLayer()
}

/**
 * Add a layer to the network.
 *
 * Validates layer connectivity and neuron counts according to requirements:
 * - Requirement 1.2: Validate neuron count > 0
 * - Requirement 1.4: Automatically connect layers in sequence
 *
 * @param input_size Number of inputs to this layer
 * @param output_size Number of outputs (neurons) in this layer
 * @param activation Shared pointer to activation function
 * @throws std::invalid_argument if validation fails
 */
void Network::addLayer(size_t input_size, size_t output_size,
                       std::shared_ptr<ActivationFunction> activation) {
    // Requirement 1.2: Validate neuron count > 0
    if (input_size == 0) {
        throw std::invalid_argument(
            "Network addLayer: input_size must be greater than zero"
        );
    }
    if (output_size == 0) {
        throw std::invalid_argument(
            "Network addLayer: output_size must be greater than zero"
        );
    }

    // Validate activation function
    if (!activation) {
        throw std::invalid_argument(
            "Network addLayer: activation function cannot be null"
        );
    }

    // Requirement 1.4: Validate layer connectivity
    // If this is not the first layer, ensure input_size matches previous layer's output_size
    if (!layers_.empty()) {
        size_t prev_output_size = layers_.back().outputSize();
        if (input_size != prev_output_size) {
            throw std::invalid_argument(
                "Network addLayer: layer connectivity validation failed. "
                "Previous layer output size is " + std::to_string(prev_output_size) +
                ", but new layer input size is " + std::to_string(input_size) +
                ". Layer input size must match previous layer output size."
            );
        }
    }

    // Create and add the layer
    layers_.emplace_back(input_size, output_size, activation);
}

/**
 * Perform forward propagation through the entire network.
 *
 * Implements:
 * - Requirement 4.1: Execute forward pass through all layers
 * - Requirement 4.5: Return output of final layer
 * - Requirement 4.6: Validate input dimensions match network input layer
 *
 * @param input Input vector (size must match network input layer)
 * @return Output vector from final layer
 * @throws std::invalid_argument if validation fails
 */
Vector Network::predict(const Vector& input) {
    // Validate network has at least one layer
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network predict: cannot predict with empty network. "
            "Add at least one layer using addLayer()."
        );
    }

    // Requirement 4.6: Validate input dimensions match network input layer
    size_t expected_input_size = layers_[0].inputSize();
    if (input.size() != expected_input_size) {
        throw std::invalid_argument(
            "Network predict: input size " + std::to_string(input.size()) +
            " does not match network input size " + std::to_string(expected_input_size)
        );
    }

    // Requirement 4.1: Execute forward pass through all layers
    Vector current_output = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        current_output = layers_[i].forward(current_output);
    }

    // Requirement 4.5: Return output of final layer
    return current_output;
}

/**
 * Get the number of layers in the network.
 *
 * Implements Requirement 1.5: Provide methods to query network structure
 *
 * @return Number of layers
 */
size_t Network::numLayers() const {
    return layers_.size();
}

/**
 * Get the complete network topology.
 *
 * Implements:
 * - Requirement 1.3: Store complete topology (input, hidden, output dimensions)
 * - Requirement 1.5: Provide methods to query network structure
 *
 * Returns a vector containing the size of each layer:
 * [input_size_of_layer_0, output_size_of_layer_0, output_size_of_layer_1, ...]
 *
 * For a network with layers [2->4, 4->3, 3->1], returns [2, 4, 3, 1]
 *
 * @return Vector of layer sizes representing the complete topology
 */
std::vector<size_t> Network::getTopology() const {
    if (layers_.empty()) {
        return {};
    }

    std::vector<size_t> topology;

    // Add input size of first layer
    topology.push_back(layers_[0].inputSize());

    // Add output size of each layer
    for (const auto& layer : layers_) {
        topology.push_back(layer.outputSize());
    }

    return topology;
}

/**
 * Get a reference to a specific layer for weight initialization or inspection.
 *
 * @param index Layer index (0-based)
 * @return Reference to the layer
 * @throws std::out_of_range if index is invalid
 */
Layer& Network::getLayer(size_t index) {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Network getLayer: index " + std::to_string(index) +
            " is out of range (network has " + std::to_string(layers_.size()) + " layers)"
        );
    }
    return layers_[index];
}

/**
 * Get a const reference to a specific layer for inspection.
 *
 * @param index Layer index (0-based)
 * @return Const reference to the layer
 * @throws std::out_of_range if index is invalid
 */
const Layer& Network::getLayer(size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Network getLayer: index " + std::to_string(index) +
            " is out of range (network has " + std::to_string(layers_.size()) + " layers)"
        );
    }
    return layers_[index];
}

/**
 * Perform backpropagation to compute gradients for all weights and biases.
 *
 * Implements the backpropagation algorithm according to Requirements 6.1-6.6:
 *
 * Algorithm:
 * 1. Compute output layer delta: δ^L = ∇_a L ⊙ σ'(z^L)
 *    where ∇_a L is the loss gradient and σ' is the activation derivative
 *
 * 2. For each hidden layer (backward from L-1 to 0):
 *    δ^l = (W^(l+1))^T * δ^(l+1) ⊙ σ'(z^l)
 *
 * 3. Compute weight gradients: ∂L/∂W^l = δ^l * (a^(l-1))^T
 *    where a^(l-1) is the input to layer l (output of layer l-1)
 *
 * 4. Compute bias gradients: ∂L/∂b^l = δ^l
 *
 * @param target Target output vector
 * @param loss_function Loss function to compute initial gradient
 * @param weight_gradients Output vector to store weight gradients (one matrix per layer)
 * @param bias_gradients Output vector to store bias gradients (one vector per layer)
 */
void Network::backpropagate(const Vector& target,
                            LossFunction& loss_function,
                            std::vector<Matrix>& weight_gradients,
                            std::vector<Vector>& bias_gradients) {
    // Validate network has layers
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network backpropagate: cannot backpropagate with empty network"
        );
    }

    size_t num_layers = layers_.size();

    // Initialize gradient storage
    weight_gradients.clear();
    bias_gradients.clear();

    // Storage for deltas (one per layer) - we'll build this backward
    std::vector<Vector> deltas;
    deltas.reserve(num_layers);

    // Step 1: Compute output layer delta
    // δ^L = ∇_a L ⊙ σ'(z^L)
    // Requirement 6.1, 6.5: Use loss gradient and activation derivative
    const Layer& output_layer = layers_[num_layers - 1];
    const Vector& output = output_layer.getLastOutput();
    const Vector& z_output = output_layer.getLastWeightedSum();  // weighted sum before activation

    // Get loss gradient: ∂L/∂a^L
    Vector loss_grad = loss_function.gradient(output, target);

    // Apply activation derivative element-wise: δ^L = ∇_a L ⊙ σ'(z^L)
    // Requirement 6.5: Apply activation function derivative
    Vector output_delta(output.size());
    std::shared_ptr<ActivationFunction> output_activation = output_layer.getActivation();
    for (size_t i = 0; i < output.size(); ++i) {
        double activation_deriv = output_activation->derivative(z_output[i]);
        output_delta[i] = loss_grad[i] * activation_deriv;
    }

    // Store output delta at the end (we'll build deltas in reverse order)
    deltas.push_back(output_delta);

    // Step 2: Propagate deltas backward through hidden layers
    // δ^l = (W^(l+1))^T * δ^(l+1) ⊙ σ'(z^l)
    // Requirement 6.3: Propagate gradients backward using chain rule
    for (int l = num_layers - 2; l >= 0; --l) {
        const Layer& current_layer = layers_[l];
        const Layer& next_layer = layers_[l + 1];
        const Vector& z_current = current_layer.getLastWeightedSum();

        // Get the delta from the next layer (which is at index 0 in our reverse-built vector)
        const Vector& next_delta = deltas[num_layers - 2 - l];

        // Compute (W^(l+1))^T * δ^(l+1)
        Matrix weights_next_transposed = next_layer.getWeights().transpose();
        Vector backprop_error = weights_next_transposed * next_delta;

        // Apply activation derivative element-wise: ⊙ σ'(z^l)
        // Requirement 6.5: Apply activation function derivative
        Vector current_delta(current_layer.outputSize());
        std::shared_ptr<ActivationFunction> current_activation = current_layer.getActivation();
        for (size_t i = 0; i < current_layer.outputSize(); ++i) {
            double activation_deriv = current_activation->derivative(z_current[i]);
            current_delta[i] = backprop_error[i] * activation_deriv;
        }

        deltas.push_back(current_delta);
    }

    // Now reverse deltas so they're in forward order (layer 0 to layer L)
    std::reverse(deltas.begin(), deltas.end());

    // Step 3 & 4: Compute weight and bias gradients for all layers
    // Requirement 6.1, 6.2: Compute gradients for all weights and biases
    for (size_t l = 0; l < num_layers; ++l) {
        const Layer& layer = layers_[l];
        const Vector& layer_input = layer.getLastInput();

        // Requirement 6.4: Use intermediate outputs from forward pass

        // Compute weight gradients: ∂L/∂W^l = δ^l * (a^(l-1))^T
        // This is an outer product: each element (i,j) = δ^l[i] * a^(l-1)[j]
        size_t output_size = layer.outputSize();
        size_t input_size = layer.inputSize();
        Matrix weight_grad(output_size, input_size);

        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                weight_grad(i, j) = deltas[l][i] * layer_input[j];
            }
        }

        weight_gradients.push_back(weight_grad);

        // Compute bias gradients: ∂L/∂b^l = δ^l
        bias_gradients.push_back(deltas[l]);
    }

    // Requirement 6.6: All gradients are now stored in weight_gradients and bias_gradients
}
