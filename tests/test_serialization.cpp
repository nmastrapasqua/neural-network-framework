#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <limits>
#include "../include/network.h"
#include "../include/serializer.h"
#include "../include/activation.h"

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

// Test 1: Basic serialization and deserialization round-trip
void testBasicRoundTrip() {
    std::cout << "Test 1: Basic round-trip serialization/deserialization..." << std::endl;

    // Create a simple network
    Network original;
    original.addLayer(2, 3, std::make_shared<Sigmoid>());
    original.addLayer(3, 1, std::make_shared<Tanh>());

    // Serialize to string stream
    std::stringstream ss;
    Serializer::serialize(original, ss);

    // Deserialize from string stream
    Network loaded = Serializer::deserialize(ss);

    // Verify topology matches
    assert(loaded.numLayers() == original.numLayers());

    std::vector<size_t> orig_topology = original.getTopology();
    std::vector<size_t> loaded_topology = loaded.getTopology();
    assert(orig_topology.size() == loaded_topology.size());
    for (size_t i = 0; i < orig_topology.size(); ++i) {
        assert(orig_topology[i] == loaded_topology[i]);
    }

    // Verify weights and biases match
    for (size_t i = 0; i < original.numLayers(); ++i) {
        const Layer& orig_layer = original.getLayer(i);
        const Layer& loaded_layer = loaded.getLayer(i);

        // Check activation function
        assert(orig_layer.activationName() == loaded_layer.activationName());

        // Check weights
        const Matrix& orig_weights = orig_layer.getWeights();
        const Matrix& loaded_weights = loaded_layer.getWeights();
        assert(orig_weights.rows() == loaded_weights.rows());
        assert(orig_weights.cols() == loaded_weights.cols());

        for (size_t r = 0; r < orig_weights.rows(); ++r) {
            for (size_t c = 0; c < orig_weights.cols(); ++c) {
                assert(approxEqual(orig_weights(r, c), loaded_weights(r, c)));
            }
        }

        // Check biases
        const Vector& orig_biases = orig_layer.getBiases();
        const Vector& loaded_biases = loaded_layer.getBiases();
        assert(orig_biases.size() == loaded_biases.size());

        for (size_t j = 0; j < orig_biases.size(); ++j) {
            assert(approxEqual(orig_biases[j], loaded_biases[j]));
        }
    }

    std::cout << "  PASSED: Network topology, weights, and biases match after round-trip" << std::endl;
}

// Test 2: Verify predictions match after deserialization
void testPredictionConsistency() {
    std::cout << "Test 2: Prediction consistency after deserialization..." << std::endl;

    // Create and initialize a network
    Network original;
    original.addLayer(2, 4, std::make_shared<Sigmoid>());
    original.addLayer(4, 1, std::make_shared<Sigmoid>());

    // Create test input
    Vector input(2);
    input[0] = 0.5;
    input[1] = 0.8;

    // Get prediction from original network
    Vector orig_output = original.predict(input);

    // Serialize and deserialize
    std::stringstream ss;
    Serializer::serialize(original, ss);
    Network loaded = Serializer::deserialize(ss);

    // Get prediction from loaded network
    Vector loaded_output = loaded.predict(input);

    // Verify outputs match
    assert(orig_output.size() == loaded_output.size());
    for (size_t i = 0; i < orig_output.size(); ++i) {
        assert(approxEqual(orig_output[i], loaded_output[i]));
    }

    std::cout << "  PASSED: Predictions match after deserialization" << std::endl;
}

// Test 3: Invalid format detection - missing header
void testInvalidHeader() {
    std::cout << "Test 3: Invalid header detection..." << std::endl;

    std::stringstream ss;
    ss << "INVALID_HEADER\n";
    ss << "LAYERS 1\n";

    bool caught_exception = false;
    try {
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
        std::string msg = e.what();
        assert(msg.find("NEURAL_NETWORK_V1") != std::string::npos);
    }

    assert(caught_exception);
    std::cout << "  PASSED: Invalid header detected" << std::endl;
}

// Test 4: Invalid format detection - missing END marker
void testMissingEnd() {
    std::cout << "Test 4: Missing END marker detection..." << std::endl;

    std::stringstream ss;
    ss << "NEURAL_NETWORK_V1\n";
    ss << "LAYERS 1\n";
    ss << "LAYER 0\n";
    ss << "  INPUT_SIZE 2\n";
    ss << "  OUTPUT_SIZE 1\n";
    ss << "  ACTIVATION sigmoid\n";
    ss << "  WEIGHTS 1 2\n";
    ss << "    0.5 0.3\n";
    ss << "  BIASES 1\n";
    ss << "    0.1\n";
    // Missing END marker

    bool caught_exception = false;
    try {
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
    }

    assert(caught_exception);
    std::cout << "  PASSED: Missing END marker detected" << std::endl;
}

// Test 5: Invalid activation function name
void testInvalidActivation() {
    std::cout << "Test 5: Invalid activation function detection..." << std::endl;

    std::stringstream ss;
    ss << "NEURAL_NETWORK_V1\n";
    ss << "LAYERS 1\n";
    ss << "LAYER 0\n";
    ss << "  INPUT_SIZE 2\n";
    ss << "  OUTPUT_SIZE 1\n";
    ss << "  ACTIVATION invalid_activation\n";
    ss << "  WEIGHTS 1 2\n";
    ss << "    0.5 0.3\n";
    ss << "  BIASES 1\n";
    ss << "    0.1\n";
    ss << "END\n";

    bool caught_exception = false;
    try {
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg = e.what();
        assert(msg.find("invalid_activation") != std::string::npos);
    }

    assert(caught_exception);
    std::cout << "  PASSED: Invalid activation function detected" << std::endl;
}

// Test 6: Dimension mismatch detection
void testDimensionMismatch() {
    std::cout << "Test 6: Dimension mismatch detection..." << std::endl;

    std::stringstream ss;
    ss << "NEURAL_NETWORK_V1\n";
    ss << "LAYERS 1\n";
    ss << "LAYER 0\n";
    ss << "  INPUT_SIZE 2\n";
    ss << "  OUTPUT_SIZE 3\n";
    ss << "  ACTIVATION sigmoid\n";
    ss << "  WEIGHTS 3 2\n";
    ss << "    0.5 0.3\n";
    ss << "    0.2 0.4\n";
    ss << "    0.1 0.6\n";
    ss << "  BIASES 2\n";  // Wrong size - should be 3
    ss << "    0.1 0.2\n";
    ss << "END\n";

    bool caught_exception = false;
    try {
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg = e.what();
        assert(msg.find("biases size") != std::string::npos);
    }

    assert(caught_exception);
    std::cout << "  PASSED: Dimension mismatch detected" << std::endl;
}

// Test 7: NaN value detection
void testNaNDetection() {
    std::cout << "Test 7: NaN value detection..." << std::endl;

    // Create a network with manually injected NaN
    Network original;
    original.addLayer(2, 1, std::make_shared<Sigmoid>());

    // Manually inject NaN into the weights
    Layer& layer = original.getLayer(0);
    Matrix& weights = layer.getWeights();
    weights(0, 0) = std::numeric_limits<double>::quiet_NaN();

    // Serialize the network with NaN
    std::stringstream ss;
    Serializer::serialize(original, ss);

    // Try to deserialize - should detect NaN
    bool caught_exception = false;
    try {
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg = e.what();
        assert(msg.find("NaN") != std::string::npos);
    } catch (const std::runtime_error& e) {
        // Stream might fail to parse NaN, which is also acceptable
        caught_exception = true;
    }

    assert(caught_exception);
    std::cout << "  PASSED: NaN value detected" << std::endl;
}

// Test 8: Multi-layer network with different activations
void testMultiLayerDifferentActivations() {
    std::cout << "Test 8: Multi-layer network with different activations..." << std::endl;

    // Create a network with different activation functions
    Network original;
    original.addLayer(3, 5, std::make_shared<ReLU>());
    original.addLayer(5, 4, std::make_shared<Tanh>());
    original.addLayer(4, 2, std::make_shared<Sigmoid>());

    // Serialize and deserialize
    std::stringstream ss;
    Serializer::serialize(original, ss);
    Network loaded = Serializer::deserialize(ss);

    // Verify all layers have correct activations
    assert(loaded.getLayer(0).activationName() == "relu");
    assert(loaded.getLayer(1).activationName() == "tanh");
    assert(loaded.getLayer(2).activationName() == "sigmoid");

    std::cout << "  PASSED: Multi-layer network with different activations" << std::endl;
}

// Test 9: Layer connectivity validation
void testLayerConnectivityValidation() {
    std::cout << "Test 9: Layer connectivity validation..." << std::endl;

    std::stringstream ss;
    ss << "NEURAL_NETWORK_V1\n";
    ss << "LAYERS 2\n";
    ss << "LAYER 0\n";
    ss << "  INPUT_SIZE 2\n";
    ss << "  OUTPUT_SIZE 3\n";
    ss << "  ACTIVATION sigmoid\n";
    ss << "  WEIGHTS 3 2\n";
    ss << "    0.5 0.3\n";
    ss << "    0.2 0.4\n";
    ss << "    0.1 0.6\n";
    ss << "  BIASES 3\n";
    ss << "    0.1 0.2 0.3\n";
    ss << "LAYER 1\n";
    ss << "  INPUT_SIZE 5\n";  // Wrong - should be 3 to match previous layer output
    ss << "  OUTPUT_SIZE 1\n";
    ss << "  ACTIVATION tanh\n";
    ss << "  WEIGHTS 1 5\n";
    ss << "    0.1 0.2 0.3 0.4 0.5\n";
    ss << "  BIASES 1\n";
    ss << "    0.1\n";
    ss << "END\n";

    bool caught_exception = false;
    try {
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::string msg = e.what();
        assert(msg.find("does not match previous layer output size") != std::string::npos);
    }

    assert(caught_exception);
    std::cout << "  PASSED: Layer connectivity validation works" << std::endl;
}

int main() {
    std::cout << "=== Serialization/Deserialization Tests ===" << std::endl;
    std::cout << std::endl;

    try {
        testBasicRoundTrip();
        testPredictionConsistency();
        testInvalidHeader();
        testMissingEnd();
        testInvalidActivation();
        testDimensionMismatch();
        testNaNDetection();
        testMultiLayerDifferentActivations();
        testLayerConnectivityValidation();

        std::cout << std::endl;
        std::cout << "=== All Tests PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << std::endl;
        std::cerr << "TEST FAILED: " << e.what() << std::endl;
        return 1;
    }
}
