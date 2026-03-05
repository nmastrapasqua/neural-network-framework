#include "network.h"
#include <stdexcept>

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
