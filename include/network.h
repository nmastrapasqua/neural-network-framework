#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <memory>
#include <cstddef>
#include "layer.h"
#include "vector.h"
#include "activation.h"

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

private:
    std::vector<Layer> layers_;
};

#endif // NETWORK_H
