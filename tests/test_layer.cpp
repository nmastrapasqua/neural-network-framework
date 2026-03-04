#include "layer.h"
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
 * Test: Forward pass with known values
 * Validates: Requirements 4.1, 4.2, 4.3
 *
 * Tests that the layer correctly computes z = W*x + b and applies activation.
 */
void testForwardPassWithKnownValues() {
    std::cout << "Testing forward pass with known values..." << std::endl;

    // Create a simple 2-input, 2-output layer with sigmoid activation
    auto sigmoid = std::make_shared<Sigmoid>();
    Layer layer(2, 2, sigmoid);

    // Set known weights and biases
    Matrix& weights = layer.getWeights();
    weights(0, 0) = 0.5;  weights(0, 1) = -0.3;
    weights(1, 0) = 0.2;  weights(1, 1) = 0.8;

    Vector& biases = layer.getBiases();
    biases[0] = 0.1;
    biases[1] = -0.2;

    // Test with known input
    Vector input{1.0, 2.0};
    Vector output = layer.forward(input);

    // Expected weighted sums:
    // z[0] = 0.5*1.0 + (-0.3)*2.0 + 0.1 = 0.5 - 0.6 + 0.1 = 0.0
    // z[1] = 0.2*1.0 + 0.8*2.0 + (-0.2) = 0.2 + 1.6 - 0.2 = 1.6

    // Expected outputs (sigmoid):
    // output[0] = sigmoid(0.0) = 0.5
    // output[1] = sigmoid(1.6) ≈ 0.8320

    assert(output.size() == 2);
    assert(approxEqual(output[0], 0.5, 1e-6));
    assert(approxEqual(output[1], 0.8320183851339245, 1e-6));

    std::cout << "  ✓ Forward pass with known values passed" << std::endl;
}

/**
 * Test: Forward pass with different activation functions
 * Validates: Requirements 2.4, 2.5, 4.3
 */
void testForwardPassWithDifferentActivations() {
    std::cout << "Testing forward pass with different activations..." << std::endl;

    // Test with Tanh
    auto tanh = std::make_shared<Tanh>();
    Layer layer_tanh(2, 1, tanh);
    Matrix& weights_tanh = layer_tanh.getWeights();
    weights_tanh(0, 0) = 1.0;
    weights_tanh(0, 1) = 1.0;
    Vector& biases_tanh = layer_tanh.getBiases();
    biases_tanh[0] = 0.0;

    Vector input{0.5, 0.5};
    Vector output_tanh = layer_tanh.forward(input);
    // z = 1.0*0.5 + 1.0*0.5 + 0.0 = 1.0
    // output = tanh(1.0) ≈ 0.7616
    assert(approxEqual(output_tanh[0], std::tanh(1.0), 1e-6));

    // Test with ReLU
    auto relu = std::make_shared<ReLU>();
    Layer layer_relu(2, 2, relu);
    Matrix& weights_relu = layer_relu.getWeights();
    weights_relu(0, 0) = 1.0;  weights_relu(0, 1) = -1.0;
    weights_relu(1, 0) = -1.0; weights_relu(1, 1) = 1.0;
    Vector& biases_relu = layer_relu.getBiases();
    biases_relu[0] = 0.5;
    biases_relu[1] = -0.5;

    Vector input2{1.0, 2.0};
    Vector output_relu = layer_relu.forward(input2);
    // z[0] = 1.0*1.0 + (-1.0)*2.0 + 0.5 = 1.0 - 2.0 + 0.5 = -0.5
    // z[1] = (-1.0)*1.0 + 1.0*2.0 + (-0.5) = -1.0 + 2.0 - 0.5 = 0.5
    // output[0] = relu(-0.5) = 0.0
    // output[1] = relu(0.5) = 0.5
    assert(approxEqual(output_relu[0], 0.0, 1e-6));
    assert(approxEqual(output_relu[1], 0.5, 1e-6));

    std::cout << "  ✓ Forward pass with different activations passed" << std::endl;
}

/**
 * Test: Xavier initialization bounds
 * Validates: Requirements 3.1, 3.2
 *
 * Tests that Xavier initialization produces weights within expected bounds.
 */
void testXavierInitialization() {
    std::cout << "Testing Xavier initialization..." << std::endl;

    auto sigmoid = std::make_shared<Sigmoid>();
    Layer layer(10, 5, sigmoid);

    // Initialize with Xavier
    layer.initializeXavier(10, 5);

    // Expected limit: sqrt(6 / (10 + 5)) = sqrt(6/15) = sqrt(0.4) ≈ 0.6325
    double expected_limit = std::sqrt(6.0 / 15.0);

    // Check all weights are within bounds
    const Matrix& weights = layer.getWeights();
    bool all_within_bounds = true;
    bool has_variation = false;
    double first_weight = weights(0, 0);

    for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
            double w = weights(i, j);
            if (w < -expected_limit || w > expected_limit) {
                all_within_bounds = false;
            }
            if (!approxEqual(w, first_weight)) {
                has_variation = true;
            }
        }
    }

    assert(all_within_bounds && "All weights should be within Xavier bounds");
    assert(has_variation && "Weights should have variation (not all identical)");

    // Test that biases remain zero after weight initialization
    const Vector& biases = layer.getBiases();
    for (size_t i = 0; i < biases.size(); ++i) {
        assert(approxEqual(biases[i], 0.0));
    }

    std::cout << "  ✓ Xavier initialization passed" << std::endl;
}

/**
 * Test: He initialization bounds
 * Validates: Requirements 3.1, 3.3
 *
 * Tests that He initialization produces weights within expected bounds.
 */
void testHeInitialization() {
    std::cout << "Testing He initialization..." << std::endl;

    auto relu = std::make_shared<ReLU>();
    Layer layer(20, 10, relu);

    // Initialize with He
    layer.initializeHe(20);

    // Expected limit: sqrt(2 / 20) = sqrt(0.1) ≈ 0.3162
    double expected_limit = std::sqrt(2.0 / 20.0);

    // Check all weights are within bounds
    const Matrix& weights = layer.getWeights();
    bool all_within_bounds = true;
    bool has_variation = false;
    double first_weight = weights(0, 0);

    for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
            double w = weights(i, j);
            if (w < -expected_limit || w > expected_limit) {
                all_within_bounds = false;
            }
            if (!approxEqual(w, first_weight)) {
                has_variation = true;
            }
        }
    }

    assert(all_within_bounds && "All weights should be within He bounds");
    assert(has_variation && "Weights should have variation (not all identical)");

    // Test that biases remain zero after weight initialization
    const Vector& biases = layer.getBiases();
    for (size_t i = 0; i < biases.size(); ++i) {
        assert(approxEqual(biases[i], 0.0));
    }

    std::cout << "  ✓ He initialization passed" << std::endl;
}

/**
 * Test: Input dimension validation
 * Validates: Requirement 4.6
 *
 * Tests that the layer rejects inputs with incorrect dimensions.
 */
void testInputDimensionValidation() {
    std::cout << "Testing input dimension validation..." << std::endl;

    auto sigmoid = std::make_shared<Sigmoid>();
    Layer layer(3, 2, sigmoid);

    // Valid input (size 3)
    Vector valid_input{1.0, 2.0, 3.0};
    Vector output = layer.forward(valid_input);
    assert(output.size() == 2);

    // Invalid input (size 2, expected 3)
    Vector invalid_input{1.0, 2.0};
    try {
        layer.forward(invalid_input);
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected - verify error message contains useful information
        std::string msg(e.what());
        assert(msg.find("input size") != std::string::npos);
        assert(msg.find("2") != std::string::npos);
        assert(msg.find("3") != std::string::npos);
    }

    // Invalid input (size 4, expected 3)
    Vector invalid_input2{1.0, 2.0, 3.0, 4.0};
    try {
        layer.forward(invalid_input2);
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Input dimension validation passed" << std::endl;
}

/**
 * Test: Layer construction validation
 * Validates: Requirements 1.2, 3.4
 */
void testLayerConstruction() {
    std::cout << "Testing layer construction..." << std::endl;

    auto sigmoid = std::make_shared<Sigmoid>();

    // Valid construction
    Layer layer(5, 3, sigmoid);
    assert(layer.inputSize() == 5);
    assert(layer.outputSize() == 3);
    assert(layer.activationName() == "sigmoid");

    // Check biases initialized to zero (Requirement 3.4)
    const Vector& biases = layer.getBiases();
    for (size_t i = 0; i < biases.size(); ++i) {
        assert(approxEqual(biases[i], 0.0));
    }

    // Test zero input size
    try {
        Layer invalid_layer(0, 3, sigmoid);
        assert(false && "Should have thrown exception for zero input size");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test zero output size
    try {
        Layer invalid_layer(5, 0, sigmoid);
        assert(false && "Should have thrown exception for zero output size");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test null activation function
    try {
        Layer invalid_layer(5, 3, nullptr);
        assert(false && "Should have thrown exception for null activation");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Layer construction passed" << std::endl;
}

/**
 * Test: Intermediate value caching
 * Validates: Requirement 4.4
 *
 * Tests that the layer caches input and output for backpropagation.
 */
void testIntermediateValueCaching() {
    std::cout << "Testing intermediate value caching..." << std::endl;

    auto sigmoid = std::make_shared<Sigmoid>();
    Layer layer(2, 2, sigmoid);

    // Set weights and biases
    Matrix& weights = layer.getWeights();
    weights(0, 0) = 1.0; weights(0, 1) = 0.0;
    weights(1, 0) = 0.0; weights(1, 1) = 1.0;
    Vector& biases = layer.getBiases();
    biases[0] = 0.0;
    biases[1] = 0.0;

    // Perform forward pass
    Vector input{0.5, -0.5};
    Vector output = layer.forward(input);

    // Check that input is cached
    const Vector& cached_input = layer.getLastInput();
    assert(cached_input.size() == 2);
    assert(approxEqual(cached_input[0], 0.5));
    assert(approxEqual(cached_input[1], -0.5));

    // Check that output is cached
    const Vector& cached_output = layer.getLastOutput();
    assert(cached_output.size() == 2);
    assert(approxEqual(cached_output[0], output[0]));
    assert(approxEqual(cached_output[1], output[1]));

    // Check that getLastActivation returns same as getLastOutput
    const Vector& cached_activation = layer.getLastActivation();
    assert(cached_activation.size() == cached_output.size());
    for (size_t i = 0; i < cached_activation.size(); ++i) {
        assert(approxEqual(cached_activation[i], cached_output[i]));
    }

    std::cout << "  ✓ Intermediate value caching passed" << std::endl;
}

/**
 * Test: Weight initialization with custom range
 * Validates: Requirement 3.1
 */
void testCustomWeightInitialization() {
    std::cout << "Testing custom weight initialization..." << std::endl;

    auto sigmoid = std::make_shared<Sigmoid>();
    Layer layer(5, 3, sigmoid);

    // Initialize with custom range
    layer.initializeWeights(-0.5, 0.5);

    // Check all weights are within bounds
    const Matrix& weights = layer.getWeights();
    bool all_within_bounds = true;
    bool has_variation = false;
    double first_weight = weights(0, 0);

    for (size_t i = 0; i < weights.rows(); ++i) {
        for (size_t j = 0; j < weights.cols(); ++j) {
            double w = weights(i, j);
            if (w < -0.5 || w > 0.5) {
                all_within_bounds = false;
            }
            if (!approxEqual(w, first_weight)) {
                has_variation = true;
            }
        }
    }

    assert(all_within_bounds && "All weights should be within custom bounds");
    assert(has_variation && "Weights should have variation");

    // Test invalid range (min >= max)
    try {
        layer.initializeWeights(0.5, 0.5);
        assert(false && "Should have thrown exception for invalid range");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    try {
        layer.initializeWeights(1.0, 0.5);
        assert(false && "Should have thrown exception for invalid range");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Custom weight initialization passed" << std::endl;
}

/**
 * Test: Layer properties and accessors
 * Validates: Requirements 1.5, 2.4
 */
void testLayerProperties() {
    std::cout << "Testing layer properties..." << std::endl;

    auto tanh = std::make_shared<Tanh>();
    Layer layer(7, 4, tanh);

    // Test size accessors
    assert(layer.inputSize() == 7);
    assert(layer.outputSize() == 4);

    // Test activation name
    assert(layer.activationName() == "tanh");

    // Test weight matrix dimensions
    const Matrix& weights = layer.getWeights();
    assert(weights.rows() == 4);  // output_size
    assert(weights.cols() == 7);  // input_size

    // Test bias vector dimensions
    const Vector& biases = layer.getBiases();
    assert(biases.size() == 4);  // output_size

    std::cout << "  ✓ Layer properties passed" << std::endl;
}

/**
 * Test: Multiple forward passes
 * Validates: Requirements 4.1, 4.4
 *
 * Tests that multiple forward passes work correctly and update cached values.
 */
void testMultipleForwardPasses() {
    std::cout << "Testing multiple forward passes..." << std::endl;

    auto relu = std::make_shared<ReLU>();
    Layer layer(2, 1, relu);

    Matrix& weights = layer.getWeights();
    weights(0, 0) = 1.0;
    weights(0, 1) = 1.0;
    Vector& biases = layer.getBiases();
    biases[0] = 0.0;

    // First forward pass
    Vector input1{1.0, 2.0};
    Vector output1 = layer.forward(input1);
    assert(approxEqual(output1[0], 3.0));  // relu(1.0 + 2.0) = 3.0

    // Check cached input
    const Vector& cached1 = layer.getLastInput();
    assert(approxEqual(cached1[0], 1.0));
    assert(approxEqual(cached1[1], 2.0));

    // Second forward pass with different input
    Vector input2{-1.0, 0.5};
    Vector output2 = layer.forward(input2);
    assert(approxEqual(output2[0], 0.0));  // relu(-1.0 + 0.5) = relu(-0.5) = 0.0

    // Check cached input is updated
    const Vector& cached2 = layer.getLastInput();
    assert(approxEqual(cached2[0], -1.0));
    assert(approxEqual(cached2[1], 0.5));

    std::cout << "  ✓ Multiple forward passes passed" << std::endl;
}

int main() {
    std::cout << "Running Layer tests..." << std::endl;
    std::cout << "=====================" << std::endl;

    testLayerConstruction();
    testForwardPassWithKnownValues();
    testForwardPassWithDifferentActivations();
    testXavierInitialization();
    testHeInitialization();
    testCustomWeightInitialization();
    testInputDimensionValidation();
    testIntermediateValueCaching();
    testLayerProperties();
    testMultipleForwardPasses();

    std::cout << std::endl;
    std::cout << "All Layer tests passed! ✓" << std::endl;

    return 0;
}
