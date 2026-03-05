#include "network.h"
#include "activation.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <iostream>
#include <cmath>
#include <memory>

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

// Custom generator for doubles in a range
rc::Gen<double> genDoubleInRange(double min, double max) {
    return rc::gen::map(rc::gen::inRange(0, 10000), [min, max](int i) {
        return min + (max - min) * (i / 10000.0);
    });
}

// Generator for activation functions
rc::Gen<std::shared_ptr<ActivationFunction>> arbActivation() {
    return rc::gen::apply([](int choice) -> std::shared_ptr<ActivationFunction> {
        switch (choice % 3) {
            case 0: return std::make_shared<Sigmoid>();
            case 1: return std::make_shared<Tanh>();
            case 2: return std::make_shared<ReLU>();
            default: return std::make_shared<Sigmoid>();
        }
    }, rc::gen::inRange(0, 3));
}

// Generator for valid layer sizes (positive integers)
rc::Gen<size_t> arbLayerSize() {
    return rc::gen::inRange<size_t>(1, 50);
}

// Generator for network topology (list of layer sizes)
// Returns a vector of sizes where topology[0] is input size,
// topology[1..n] are output sizes of each layer
rc::Gen<std::vector<size_t>> arbTopology() {
    return rc::gen::mapcat(
        rc::gen::inRange<size_t>(2, 6),
        [](size_t num_layers) {
            // num_layers is the number of actual layers (2-5)
            // We need num_layers + 1 sizes (input + output of each layer)
            return rc::gen::container<std::vector<size_t>>(num_layers + 1, arbLayerSize());
        }
    );
}

// **Validates: Requirements 1.1, 1.5**
// Feature: neural-network-framework, Property 1: Network Creation with Arbitrary Layers
// For any positive integer N and any sequence of positive layer sizes,
// creating a network with N layers should succeed and the network should report N layers.
RC_GTEST_PROP(NetworkPropertyTest, NetworkCreationWithArbitraryLayers, ()) {
    // Generate a random topology
    auto topology = *arbTopology();
    size_t num_layers = topology.size() - 1;  // topology includes input size

    // Create network
    Network network;

    // Add layers according to topology
    for (size_t i = 0; i < num_layers; ++i) {
        auto activation = *arbActivation();
        network.addLayer(topology[i], topology[i + 1], activation);
    }

    // Verify the network reports the correct number of layers
    RC_ASSERT(network.numLayers() == num_layers);
}

// **Validates: Requirements 1.2**
// Feature: neural-network-framework, Property 2: Layer Size Validation
// For any attempt to add a layer with zero or negative neurons,
// the system should reject the operation and signal an error.
RC_GTEST_PROP(NetworkPropertyTest, LayerSizeValidation, ()) {
    Network network;
    auto activation = *arbActivation();

    // Test zero input_size
    RC_ASSERT_THROWS_AS(
        network.addLayer(0, 5, activation),
        std::invalid_argument
    );

    // Test zero output_size
    RC_ASSERT_THROWS_AS(
        network.addLayer(5, 0, activation),
        std::invalid_argument
    );

    // Test both zero
    RC_ASSERT_THROWS_AS(
        network.addLayer(0, 0, activation),
        std::invalid_argument
    );
}

// **Validates: Requirements 1.3, 1.5**
// Feature: neural-network-framework, Property 3: Topology Preservation
// For any network created with a specific topology (layer sizes and activation functions),
// querying the network's topology should return exactly the same structure
// that was specified during creation.
RC_GTEST_PROP(NetworkPropertyTest, TopologyPreservation, ()) {
    // Generate a random topology
    auto topology = *arbTopology();
    size_t num_layers = topology.size() - 1;

    // Create network and store activation names
    Network network;
    std::vector<std::string> expected_activations;

    for (size_t i = 0; i < num_layers; ++i) {
        auto activation = *arbActivation();
        expected_activations.push_back(activation->name());
        network.addLayer(topology[i], topology[i + 1], activation);
    }

    // Query the topology
    auto retrieved_topology = network.getTopology();

    // Verify topology matches
    RC_ASSERT(retrieved_topology.size() == topology.size());
    for (size_t i = 0; i < topology.size(); ++i) {
        RC_ASSERT(retrieved_topology[i] == topology[i]);
    }

    // Verify activation functions are preserved
    for (size_t i = 0; i < num_layers; ++i) {
        RC_ASSERT(network.getLayer(i).activationName() == expected_activations[i]);
    }
}

// **Validates: Requirements 1.4**
// Feature: neural-network-framework, Property 4: Layer Connectivity
// For any network with multiple layers, the output size of layer N should equal
// the input size of layer N+1 for all consecutive layers.
RC_GTEST_PROP(NetworkPropertyTest, LayerConnectivity, ()) {
    // Generate a random topology
    auto topology = *arbTopology();
    size_t num_layers = topology.size() - 1;

    // Create network
    Network network;

    for (size_t i = 0; i < num_layers; ++i) {
        auto activation = *arbActivation();
        network.addLayer(topology[i], topology[i + 1], activation);
    }

    // Verify connectivity: output_size[i] == input_size[i+1]
    for (size_t i = 0; i < num_layers - 1; ++i) {
        size_t output_size_i = network.getLayer(i).outputSize();
        size_t input_size_i_plus_1 = network.getLayer(i + 1).inputSize();
        RC_ASSERT(output_size_i == input_size_i_plus_1);
    }
}

// Additional test: Verify that adding a layer with mismatched dimensions fails
RC_GTEST_PROP(NetworkPropertyTest, LayerConnectivityValidation, ()) {
    Network network;
    auto activation = *arbActivation();

    // Add first layer
    size_t first_input = *arbLayerSize();
    size_t first_output = *arbLayerSize();
    network.addLayer(first_input, first_output, activation);

    // Try to add a second layer with mismatched input size
    size_t mismatched_input = first_output + 1;  // Intentionally wrong
    size_t second_output = *arbLayerSize();

    RC_ASSERT_THROWS_AS(
        network.addLayer(mismatched_input, second_output, activation),
        std::invalid_argument
    );
}

// **Validates: Requirements 4.1**
// Feature: neural-network-framework, Property 12: Forward Pass Execution
// For any network and any input vector of correct dimensions,
// the forward pass should complete successfully and produce an output vector
// with dimensions matching the output layer size.
RC_GTEST_PROP(NetworkPropertyTest, ForwardPassExecution, ()) {
    // Generate a random topology
    auto topology = *arbTopology();
    size_t num_layers = topology.size() - 1;

    // Create network
    Network network;

    for (size_t i = 0; i < num_layers; ++i) {
        auto activation = *arbActivation();
        network.addLayer(topology[i], topology[i + 1], activation);
    }

    // Create input vector with correct dimensions
    size_t input_size = topology[0];
    Vector input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = *genDoubleInRange(-1.0, 1.0);
    }

    // Perform forward pass
    Vector output = network.predict(input);

    // Verify output dimensions match the final layer's output size
    size_t expected_output_size = topology.back();
    RC_ASSERT(output.size() == expected_output_size);
}

// **Validates: Requirements 4.6**
// Feature: neural-network-framework, Property 15: Input Dimension Validation
// For any network with input size N, providing an input vector of size M
// where M ≠ N should be rejected with a descriptive error.
RC_GTEST_PROP(NetworkPropertyTest, InputDimensionValidation, ()) {
    // Generate a random topology
    auto topology = *arbTopology();
    size_t num_layers = topology.size() - 1;

    // Create network
    Network network;

    for (size_t i = 0; i < num_layers; ++i) {
        auto activation = *arbActivation();
        network.addLayer(topology[i], topology[i + 1], activation);
    }

    // Create input vector with WRONG dimensions
    size_t correct_input_size = topology[0];
    size_t wrong_input_size = correct_input_size + 1;
    Vector wrong_input(wrong_input_size, 0.0);

    // Verify that predict throws an exception
    RC_ASSERT_THROWS_AS(
        network.predict(wrong_input),
        std::invalid_argument
    );
}

// Additional test: Empty network should reject predict
RC_GTEST_PROP(NetworkPropertyTest, EmptyNetworkRejectsPrediction, ()) {
    Network network;

    // Try to predict with empty network
    size_t input_size = *arbLayerSize();
    Vector input(input_size, 0.0);

    RC_ASSERT_THROWS_AS(
        network.predict(input),
        std::invalid_argument
    );
}

// Additional test: Null activation function should be rejected
RC_GTEST_PROP(NetworkPropertyTest, NullActivationRejection, ()) {
    Network network;

    size_t input_size = *arbLayerSize();
    size_t output_size = *arbLayerSize();

    RC_ASSERT_THROWS_AS(
        network.addLayer(input_size, output_size, nullptr),
        std::invalid_argument
    );
}

// Additional test: Forward pass produces finite values
RC_GTEST_PROP(NetworkPropertyTest, ForwardPassProducesFiniteValues, ()) {
    // Generate a small topology to avoid numerical issues
    size_t num_layers = *rc::gen::inRange<size_t>(2, 4);
    std::vector<size_t> topology;
    topology.reserve(num_layers + 1);
    for (size_t i = 0; i <= num_layers; ++i) {
        topology.push_back(*rc::gen::inRange<size_t>(1, 10));
    }

    size_t input_size = topology[0];

    // Create network
    Network network;

    for (size_t i = 0; i < num_layers; ++i) {
        auto activation = *arbActivation();
        network.addLayer(topology[i], topology[i + 1], activation);
    }

    // Create input vector with reasonable values
    Vector input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = *genDoubleInRange(-1.0, 1.0);
    }

    // Perform forward pass
    Vector output = network.predict(input);

    // Verify all output values are finite
    for (size_t i = 0; i < output.size(); ++i) {
        RC_ASSERT(std::isfinite(output[i]));
    }
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
