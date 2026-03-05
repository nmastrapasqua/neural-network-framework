#include "network.h"
#include "activation.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <memory>

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

/**
 * Test: Network construction
 * Validates: Requirement 1.1
 *
 * Tests that an empty network can be created.
 */
void testNetworkConstruction() {
    std::cout << "Testing network construction..." << std::endl;

    Network network;
    assert(network.numLayers() == 0);

    std::vector<size_t> topology = network.getTopology();
    assert(topology.empty());

    std::cout << "  ✓ Network construction passed" << std::endl;
}

/**
 * Test: Adding single layer
 * Validates: Requirements 1.1, 1.2, 1.5
 *
 * Tests that a single layer can be added to the network.
 */
void testAddSingleLayer() {
    std::cout << "Testing adding single layer..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    network.addLayer(3, 2, sigmoid);

    assert(network.numLayers() == 1);

    std::vector<size_t> topology = network.getTopology();
    assert(topology.size() == 2);
    assert(topology[0] == 3);  // input size
    assert(topology[1] == 2);  // output size

    std::cout << "  ✓ Adding single layer passed" << std::endl;
}

/**
 * Test: Adding multiple layers
 * Validates: Requirements 1.1, 1.3, 1.4, 1.5
 *
 * Tests that multiple layers can be added with proper connectivity.
 */
void testAddMultipleLayers() {
    std::cout << "Testing adding multiple layers..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    auto tanh = std::make_shared<Tanh>();
    auto relu = std::make_shared<ReLU>();

    // Build network: 2 -> 4 -> 3 -> 1
    network.addLayer(2, 4, sigmoid);
    network.addLayer(4, 3, tanh);
    network.addLayer(3, 1, relu);

    assert(network.numLayers() == 3);

    // Requirement 1.3: Store complete topology
    std::vector<size_t> topology = network.getTopology();
    assert(topology.size() == 4);
    assert(topology[0] == 2);  // input
    assert(topology[1] == 4);  // hidden 1
    assert(topology[2] == 3);  // hidden 2
    assert(topology[3] == 1);  // output

    std::cout << "  ✓ Adding multiple layers passed" << std::endl;
}

/**
 * Test: Layer size validation
 * Validates: Requirement 1.2
 *
 * Tests that zero or negative layer sizes are rejected.
 */
void testLayerSizeValidation() {
    std::cout << "Testing layer size validation..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    // Test zero input size
    try {
        network.addLayer(0, 3, sigmoid);
        assert(false && "Should have thrown exception for zero input size");
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        assert(msg.find("input_size") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }

    // Test zero output size
    try {
        network.addLayer(3, 0, sigmoid);
        assert(false && "Should have thrown exception for zero output size");
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        assert(msg.find("output_size") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }

    // Test null activation function
    try {
        network.addLayer(3, 2, nullptr);
        assert(false && "Should have thrown exception for null activation");
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        assert(msg.find("activation") != std::string::npos);
    }

    std::cout << "  ✓ Layer size validation passed" << std::endl;
}

/**
 * Test: Layer connectivity validation
 * Validates: Requirement 1.4
 *
 * Tests that layers must be properly connected (output_size[i] == input_size[i+1]).
 */
void testLayerConnectivityValidation() {
    std::cout << "Testing layer connectivity validation..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    auto tanh = std::make_shared<Tanh>();

    // Add first layer: 3 -> 5
    network.addLayer(3, 5, sigmoid);

    // Try to add incompatible second layer: 4 -> 2 (should fail, needs 5 -> X)
    try {
        network.addLayer(4, 2, tanh);
        assert(false && "Should have thrown exception for connectivity mismatch");
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        assert(msg.find("connectivity") != std::string::npos);
        assert(msg.find("5") != std::string::npos);  // previous output size
        assert(msg.find("4") != std::string::npos);  // attempted input size
    }

    // Add compatible second layer: 5 -> 2 (should succeed)
    network.addLayer(5, 2, tanh);
    assert(network.numLayers() == 2);

    std::cout << "  ✓ Layer connectivity validation passed" << std::endl;
}

/**
 * Test: Forward pass with single layer
 * Validates: Requirements 4.1, 4.5, 4.6
 *
 * Tests forward propagation through a single-layer network.
 */
void testForwardPassSingleLayer() {
    std::cout << "Testing forward pass with single layer..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    network.addLayer(2, 1, sigmoid);

    // Manually set weights and biases for predictable output
    // (This would normally be done through training or initialization)
    // For now, we just test that predict() executes without error

    Vector input{0.5, -0.3};
    Vector output = network.predict(input);

    // Requirement 4.5: Return output of final layer
    assert(output.size() == 1);

    std::cout << "  ✓ Forward pass with single layer passed" << std::endl;
}

/**
 * Test: Forward pass with multiple layers
 * Validates: Requirements 4.1, 4.5
 *
 * Tests forward propagation through a multi-layer network.
 */
void testForwardPassMultipleLayers() {
    std::cout << "Testing forward pass with multiple layers..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    auto tanh = std::make_shared<Tanh>();

    // Build network: 2 -> 3 -> 1
    network.addLayer(2, 3, sigmoid);
    network.addLayer(3, 1, tanh);

    Vector input{1.0, -1.0};
    Vector output = network.predict(input);

    // Requirement 4.5: Return output of final layer
    assert(output.size() == 1);

    // Output should be in tanh range (-1, 1)
    assert(output[0] >= -1.0 && output[0] <= 1.0);

    std::cout << "  ✓ Forward pass with multiple layers passed" << std::endl;
}

/**
 * Test: Input dimension validation
 * Validates: Requirement 4.6
 *
 * Tests that predict() validates input dimensions.
 */
void testInputDimensionValidation() {
    std::cout << "Testing input dimension validation..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    network.addLayer(3, 2, sigmoid);

    // Valid input (size 3)
    Vector valid_input{1.0, 2.0, 3.0};
    Vector output = network.predict(valid_input);
    assert(output.size() == 2);

    // Invalid input (size 2, expected 3)
    Vector invalid_input{1.0, 2.0};
    try {
        network.predict(invalid_input);
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        assert(msg.find("input size") != std::string::npos);
        assert(msg.find("2") != std::string::npos);
        assert(msg.find("3") != std::string::npos);
    }

    // Invalid input (size 4, expected 3)
    Vector invalid_input2{1.0, 2.0, 3.0, 4.0};
    try {
        network.predict(invalid_input2);
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Input dimension validation passed" << std::endl;
}

/**
 * Test: Predict on empty network
 * Validates: Error handling
 *
 * Tests that predict() fails gracefully on empty network.
 */
void testPredictOnEmptyNetwork() {
    std::cout << "Testing predict on empty network..." << std::endl;

    Network network;
    Vector input{1.0, 2.0};

    try {
        network.predict(input);
        assert(false && "Should have thrown exception for empty network");
    } catch (const std::invalid_argument& e) {
        std::string msg(e.what());
        assert(msg.find("empty network") != std::string::npos);
    }

    std::cout << "  ✓ Predict on empty network passed" << std::endl;
}

/**
 * Test: Network topology query
 * Validates: Requirements 1.3, 1.5
 *
 * Tests that getTopology() returns correct network structure.
 */
void testNetworkTopology() {
    std::cout << "Testing network topology query..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    // Empty network
    std::vector<size_t> topology0 = network.getTopology();
    assert(topology0.empty());

    // Single layer: 5 -> 3
    network.addLayer(5, 3, sigmoid);
    std::vector<size_t> topology1 = network.getTopology();
    assert(topology1.size() == 2);
    assert(topology1[0] == 5);
    assert(topology1[1] == 3);

    // Two layers: 5 -> 3 -> 2
    network.addLayer(3, 2, sigmoid);
    std::vector<size_t> topology2 = network.getTopology();
    assert(topology2.size() == 3);
    assert(topology2[0] == 5);
    assert(topology2[1] == 3);
    assert(topology2[2] == 2);

    // Three layers: 5 -> 3 -> 2 -> 1
    network.addLayer(2, 1, sigmoid);
    std::vector<size_t> topology3 = network.getTopology();
    assert(topology3.size() == 4);
    assert(topology3[0] == 5);
    assert(topology3[1] == 3);
    assert(topology3[2] == 2);
    assert(topology3[3] == 1);

    std::cout << "  ✓ Network topology query passed" << std::endl;
}

/**
 * Test: Number of layers query
 * Validates: Requirement 1.5
 *
 * Tests that numLayers() returns correct count.
 */
void testNumLayers() {
    std::cout << "Testing number of layers query..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    assert(network.numLayers() == 0);

    network.addLayer(2, 3, sigmoid);
    assert(network.numLayers() == 1);

    network.addLayer(3, 4, sigmoid);
    assert(network.numLayers() == 2);

    network.addLayer(4, 1, sigmoid);
    assert(network.numLayers() == 3);

    std::cout << "  ✓ Number of layers query passed" << std::endl;
}

/**
 * Test: Complex network architecture
 * Validates: Requirements 1.1, 1.3, 1.4, 1.5, 4.1
 *
 * Tests a more complex network with many layers.
 */
void testComplexNetworkArchitecture() {
    std::cout << "Testing complex network architecture..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    auto tanh = std::make_shared<Tanh>();
    auto relu = std::make_shared<ReLU>();

    // Build network: 10 -> 8 -> 6 -> 4 -> 2 -> 1
    network.addLayer(10, 8, relu);
    network.addLayer(8, 6, relu);
    network.addLayer(6, 4, tanh);
    network.addLayer(4, 2, sigmoid);
    network.addLayer(2, 1, sigmoid);

    assert(network.numLayers() == 5);

    std::vector<size_t> topology = network.getTopology();
    assert(topology.size() == 6);
    assert(topology[0] == 10);
    assert(topology[1] == 8);
    assert(topology[2] == 6);
    assert(topology[3] == 4);
    assert(topology[4] == 2);
    assert(topology[5] == 1);

    // Test forward pass
    Vector input(10, 0.5);  // 10-dimensional input, all 0.5
    Vector output = network.predict(input);
    assert(output.size() == 1);

    std::cout << "  ✓ Complex network architecture passed" << std::endl;
}

/**
 * Test: Multiple predictions
 * Validates: Requirements 4.1, 4.5
 *
 * Tests that multiple predictions can be made on the same network.
 */
void testMultiplePredictions() {
    std::cout << "Testing multiple predictions..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    network.addLayer(2, 3, sigmoid);
    network.addLayer(3, 1, sigmoid);

    // Note: With zero-initialized weights, all outputs will be sigmoid(0) = 0.5
    // This test validates that predict() can be called multiple times successfully

    // First prediction
    Vector input1{0.5, 0.5};
    Vector output1 = network.predict(input1);
    assert(output1.size() == 1);

    // Second prediction with different input
    Vector input2{-0.5, 1.0};
    Vector output2 = network.predict(input2);
    assert(output2.size() == 1);

    // Third prediction
    Vector input3{0.0, 0.0};
    Vector output3 = network.predict(input3);
    assert(output3.size() == 1);

    // With zero weights, all outputs should be sigmoid(0) = 0.5
    // This validates that the network is stateless between predictions
    assert(approxEqual(output1[0], 0.5, 1e-6));
    assert(approxEqual(output2[0], 0.5, 1e-6));
    assert(approxEqual(output3[0], 0.5, 1e-6));

    std::cout << "  ✓ Multiple predictions passed" << std::endl;
}

/**
 * Test: Network with different activation functions per layer
 * Validates: Requirements 2.4, 4.1
 *
 * Tests that each layer can have a different activation function.
 */
void testDifferentActivationsPerLayer() {
    std::cout << "Testing different activations per layer..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    auto tanh = std::make_shared<Tanh>();
    auto relu = std::make_shared<ReLU>();

    // Each layer uses a different activation
    network.addLayer(3, 4, relu);
    network.addLayer(4, 3, tanh);
    network.addLayer(3, 1, sigmoid);

    Vector input{1.0, -1.0, 0.5};
    Vector output = network.predict(input);

    assert(output.size() == 1);
    // Output should be in sigmoid range (0, 1)
    assert(output[0] > 0.0 && output[0] < 1.0);

    std::cout << "  ✓ Different activations per layer passed" << std::endl;
}

/**
 * Test: Network with Xavier weight initialization
 * Validates: Requirements 3.2, 4.1
 *
 * Tests that a network can use Xavier initialization and produce varied outputs.
 */
void testNetworkWithXavierInitialization() {
    std::cout << "Testing network with Xavier initialization..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    // Build network: 2 -> 4 -> 1
    network.addLayer(2, 4, sigmoid);
    network.addLayer(4, 1, sigmoid);

    // Initialize weights using Xavier for each layer
    network.getLayer(0).initializeXavier(2, 4);
    network.getLayer(1).initializeXavier(4, 1);

    // Verify weights are in Xavier range for layer 0
    // Range: [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
    double limit0 = std::sqrt(6.0 / (2 + 4));  // sqrt(6/6) = 1.0
    const Matrix& weights0 = network.getLayer(0).getWeights();
    for (size_t i = 0; i < weights0.rows(); ++i) {
        for (size_t j = 0; j < weights0.cols(); ++j) {
            assert(weights0(i, j) >= -limit0 && weights0(i, j) <= limit0);
        }
    }

    // Verify weights are in Xavier range for layer 1
    double limit1 = std::sqrt(6.0 / (4 + 1));  // sqrt(6/5) ≈ 1.095
    const Matrix& weights1 = network.getLayer(1).getWeights();
    for (size_t i = 0; i < weights1.rows(); ++i) {
        for (size_t j = 0; j < weights1.cols(); ++j) {
            assert(weights1(i, j) >= -limit1 && weights1(i, j) <= limit1);
        }
    }

    // Test forward pass with initialized weights through the network
    Vector input{0.5, -0.3};
    Vector output = network.predict(input);
    assert(output.size() == 1);

    // With Xavier initialization, output should not be exactly 0.5 (which would be the case with zero weights)
    assert(!approxEqual(output[0], 0.5, 0.01));

    // Test multiple predictions with different inputs
    Vector input2{1.0, 1.0};
    Vector output2 = network.predict(input2);
    assert(output2.size() == 1);

    Vector input3{-1.0, -1.0};
    Vector output3 = network.predict(input3);
    assert(output3.size() == 1);

    // Outputs should vary for different inputs
    bool has_variation = !approxEqual(output[0], output2[0], 0.01) ||
                         !approxEqual(output2[0], output3[0], 0.01);
    assert(has_variation);

    std::cout << "  ✓ Network with Xavier initialization passed" << std::endl;
}

/**
 * Test: Network with manual weight initialization
 * Validates: Requirements 3.1, 4.1
 *
 * Tests that weights and biases can be set manually to arbitrary values.
 */
void testNetworkWithManualWeightInitialization() {
    std::cout << "Testing network with manual weight initialization..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();

    // Build simple network: 2 -> 1
    network.addLayer(2, 1, sigmoid);

    // Manually set specific weight values
    Matrix& weights = network.getLayer(0).getWeights();
    weights(0, 0) = 0.5;   // weight from input 0 to output 0
    weights(0, 1) = -0.3;  // weight from input 1 to output 0

    // Manually set bias
    Vector& biases = network.getLayer(0).getBiases();
    biases[0] = 0.1;

    // Verify weights were set correctly
    assert(approxEqual(weights(0, 0), 0.5));
    assert(approxEqual(weights(0, 1), -0.3));
    assert(approxEqual(biases[0], 0.1));

    // Test forward pass with known weights
    // z = 0.5*1.0 + (-0.3)*2.0 + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
    // output = sigmoid(0.0) = 0.5
    Vector input{1.0, 2.0};
    Vector output = network.predict(input);

    assert(output.size() == 1);
    assert(approxEqual(output[0], 0.5, 1e-6));

    // Test with different input
    // z = 0.5*2.0 + (-0.3)*1.0 + 0.1 = 1.0 - 0.3 + 0.1 = 0.8
    // output = sigmoid(0.8) ≈ 0.689
    Vector input2{2.0, 1.0};
    Vector output2 = network.predict(input2);

    double expected = 1.0 / (1.0 + std::exp(-0.8));  // sigmoid(0.8)
    assert(approxEqual(output2[0], expected, 1e-6));

    std::cout << "  ✓ Network with manual weight initialization passed" << std::endl;
}

/**
 * Test: Network with mixed initialization strategies
 * Validates: Requirements 3.1, 3.2, 3.3
 *
 * Tests that different layers can use different initialization strategies.
 */
void testNetworkWithMixedInitialization() {
    std::cout << "Testing network with mixed initialization strategies..." << std::endl;

    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    auto relu = std::make_shared<ReLU>();

    // Build network: 3 -> 4 -> 2
    network.addLayer(3, 4, relu);
    network.addLayer(4, 2, sigmoid);

    // Layer 0: Use He initialization (good for ReLU)
    network.getLayer(0).initializeHe(3);

    // Layer 1: Use Xavier initialization (good for sigmoid)
    network.getLayer(1).initializeXavier(4, 2);

    // Verify He initialization range for layer 0: [-sqrt(2/fan_in), sqrt(2/fan_in)]
    double he_limit = std::sqrt(2.0 / 3);
    const Matrix& weights0 = network.getLayer(0).getWeights();
    for (size_t i = 0; i < weights0.rows(); ++i) {
        for (size_t j = 0; j < weights0.cols(); ++j) {
            assert(weights0(i, j) >= -he_limit && weights0(i, j) <= he_limit);
        }
    }

    // Verify Xavier initialization range for layer 1: [-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out))]
    double xavier_limit = std::sqrt(6.0 / (4 + 2));
    const Matrix& weights1 = network.getLayer(1).getWeights();
    for (size_t i = 0; i < weights1.rows(); ++i) {
        for (size_t j = 0; j < weights1.cols(); ++j) {
            assert(weights1(i, j) >= -xavier_limit && weights1(i, j) <= xavier_limit);
        }
    }

    // Test forward pass
    Vector input{0.5, -0.5, 1.0};
    Vector output = network.predict(input);
    assert(output.size() == 2);

    // Outputs should be in sigmoid range (0, 1) since last layer uses sigmoid
    assert(output[0] > 0.0 && output[0] < 1.0);
    assert(output[1] > 0.0 && output[1] < 1.0);

    std::cout << "  ✓ Network with mixed initialization strategies passed" << std::endl;
}

int main() {
    std::cout << "Running Network tests..." << std::endl;
    std::cout << "=======================" << std::endl;

    testNetworkConstruction();
    testAddSingleLayer();
    testAddMultipleLayers();
    testLayerSizeValidation();
    testLayerConnectivityValidation();
    testForwardPassSingleLayer();
    testForwardPassMultipleLayers();
    testInputDimensionValidation();
    testPredictOnEmptyNetwork();
    testNetworkTopology();
    testNumLayers();
    testComplexNetworkArchitecture();
    testMultiplePredictions();
    testDifferentActivationsPerLayer();
    testNetworkWithXavierInitialization();
    testNetworkWithManualWeightInitialization();
    testNetworkWithMixedInitialization();

    std::cout << std::endl;
    std::cout << "All Network tests passed! ✓" << std::endl;

    return 0;
}
