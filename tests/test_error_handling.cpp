/**
 * Unit tests for error handling and validation (Task 15.1)
 *
 * Tests Requirements 12.1-12.5:
 * - 12.1: NaN/Inf detection in forward pass, loss computation, backpropagation
 * - 12.2: Configuration parameter validation
 * - 12.3: Dimension validation
 * - 12.4: Memory allocation error handling (handled by RAII)
 * - 12.5: Descriptive error messages with context
 */

#include "network.h"
#include "layer.h"
#include "activation.h"
#include "loss.h"
#include "matrix.h"
#include "vector.h"
#include "validation.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// Test 1: Configuration parameter validation
void testConfigurationValidation() {
    std::cout << "Test 1: Configuration parameter validation..." << std::endl;

    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());

    // Initialize weights
    net.getLayer(0).initializeXavier(2, 3);
    net.getLayer(1).initializeXavier(3, 1);

    std::vector<Vector> inputs = {{0.0, 0.0}};
    std::vector<Vector> targets = {{0.0}};
    MeanSquaredError mse;

    // Test 1.1: Zero epochs
    bool caught = false;
    try {
        net.train(inputs, targets, 0, 0.1, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("epochs") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Zero epochs rejected" << std::endl;

    // Test 1.2: Negative learning rate
    caught = false;
    try {
        net.train(inputs, targets, 1, -0.1, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("learning_rate") != std::string::npos);
        assert(msg.find("positive") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Negative learning rate rejected" << std::endl;

    // Test 1.3: Zero learning rate
    caught = false;
    try {
        net.train(inputs, targets, 1, 0.0, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("learning_rate") != std::string::npos);
        assert(msg.find("positive") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Zero learning rate rejected" << std::endl;

    // Test 1.4: Zero batch size
    caught = false;
    try {
        net.train(inputs, targets, 1, 0.1, mse, 0);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("batch_size") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Zero batch size rejected" << std::endl;

    // Test 1.5: Batch size larger than dataset
    caught = false;
    try {
        net.train(inputs, targets, 1, 0.1, mse, 10);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("batch_size") != std::string::npos);
        assert(msg.find("dataset size") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Batch size > dataset size rejected" << std::endl;

    // Test 1.6: Zero layer size
    caught = false;
    try {
        Network net2;
        net2.addLayer(0, 3, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("input_size") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Zero input size rejected" << std::endl;

    caught = false;
    try {
        Network net2;
        net2.addLayer(2, 0, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("output_size") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Zero output size rejected" << std::endl;

    std::cout << "  PASSED: Configuration validation" << std::endl;
}

// Test 2: Dimension mismatch error messages
void testDimensionMismatchMessages() {
    std::cout << "\nTest 2: Dimension mismatch error messages..." << std::endl;

    // Test 2.1: Input dimension mismatch
    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);

    bool caught = false;
    try {
        Vector wrong_input = {1.0, 2.0, 3.0};  // 3 elements instead of 2
        net.predict(wrong_input);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("input size") != std::string::npos);
        assert(msg.find("3") != std::string::npos);  // actual size
        assert(msg.find("2") != std::string::npos);  // expected size
    }
    assert(caught);
    std::cout << "  ✓ Input dimension mismatch detected with context" << std::endl;

    // Test 2.2: Vector operation dimension mismatch
    caught = false;
    try {
        Vector v1 = {1.0, 2.0};
        Vector v2 = {1.0, 2.0, 3.0};
        Vector result = v1 + v2;
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("addition") != std::string::npos);
        assert(msg.find("dimensions mismatch") != std::string::npos);
        assert(msg.find("2") != std::string::npos);
        assert(msg.find("3") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Vector dimension mismatch detected with context" << std::endl;

    // Test 2.3: Matrix operation dimension mismatch
    caught = false;
    try {
        Matrix m1(2, 3);
        Matrix m2(4, 2);
        Matrix result = m1 * m2;
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("multiplication") != std::string::npos);
        assert(msg.find("incompatible") != std::string::npos);
        assert(msg.find("2x3") != std::string::npos);
        assert(msg.find("4x2") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Matrix dimension mismatch detected with context" << std::endl;

    std::cout << "  PASSED: Dimension mismatch messages" << std::endl;
}

// Test 3: NaN detection in forward pass
void testNaNDetectionForwardPass() {
    std::cout << "\nTest 3: NaN detection in forward pass..." << std::endl;

    Network net;
    net.addLayer(2, 2, std::make_shared<Sigmoid>());

    // Manually inject NaN into weights
    Layer& layer = net.getLayer(0);
    Matrix& weights = layer.getWeights();
    weights(0, 0) = std::numeric_limits<double>::quiet_NaN();

    bool caught = false;
    try {
        Vector input = {1.0, 1.0};
        net.predict(input);
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("NaN") != std::string::npos);
        assert(msg.find("weighted sum") != std::string::npos ||
               msg.find("activation") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ NaN detected in forward pass with context" << std::endl;

    std::cout << "  PASSED: NaN detection in forward pass" << std::endl;
}

// Test 4: Inf detection in forward pass
void testInfDetectionForwardPass() {
    std::cout << "\nTest 4: Inf detection in forward pass..." << std::endl;

    Network net;
    net.addLayer(2, 2, std::make_shared<Sigmoid>());

    // Manually inject Inf into weights
    Layer& layer = net.getLayer(0);
    Matrix& weights = layer.getWeights();
    weights(0, 0) = std::numeric_limits<double>::infinity();

    bool caught = false;
    try {
        Vector input = {1.0, 1.0};
        net.predict(input);
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("Inf") != std::string::npos);
        assert(msg.find("weighted sum") != std::string::npos ||
               msg.find("activation") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ Inf detected in forward pass with context" << std::endl;

    std::cout << "  PASSED: Inf detection in forward pass" << std::endl;
}

// Test 5: NaN detection in loss computation
void testNaNDetectionLoss() {
    std::cout << "\nTest 5: NaN detection in loss computation..." << std::endl;

    // Create vectors that would produce NaN in loss computation
    // For MSE: if predicted or target contains NaN
    Vector predicted = {std::numeric_limits<double>::quiet_NaN(), 0.5};
    Vector target = {0.5, 0.5};

    MeanSquaredError mse;

    bool caught = false;
    try {
        double loss = mse.compute(predicted, target);
        (void)loss;  // Suppress unused variable warning
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("NaN") != std::string::npos);
        assert(msg.find("MSE") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ NaN detected in MSE loss computation" << std::endl;

    std::cout << "  PASSED: NaN detection in loss" << std::endl;
}

// Test 6: NaN detection in backpropagation
void testNaNDetectionBackprop() {
    std::cout << "\nTest 6: NaN detection in backpropagation..." << std::endl;

    Network net;
    net.addLayer(2, 2, std::make_shared<Sigmoid>());
    net.addLayer(2, 1, std::make_shared<Sigmoid>());

    // Initialize with normal weights
    net.getLayer(0).initializeXavier(2, 2);
    net.getLayer(1).initializeXavier(2, 1);

    // Inject NaN into a weight that will affect backprop
    Layer& layer = net.getLayer(1);
    Matrix& weights = layer.getWeights();
    weights(0, 0) = std::numeric_limits<double>::quiet_NaN();

    std::vector<Vector> inputs = {{1.0, 1.0}};
    std::vector<Vector> targets = {{0.5}};
    MeanSquaredError mse;

    bool caught = false;
    try {
        // Training will trigger backpropagation
        net.train(inputs, targets, 1, 0.1, mse);
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        // Should catch NaN either in forward pass or backprop
        assert(msg.find("NaN") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ NaN detected during training (forward/backprop)" << std::endl;

    std::cout << "  PASSED: NaN detection in backpropagation" << std::endl;
}

// Test 7: Validation utility functions
void testValidationUtilities() {
    std::cout << "\nTest 7: Validation utility functions..." << std::endl;

    // Test validateFinite with NaN
    bool caught = false;
    try {
        Validation::validateFinite(std::numeric_limits<double>::quiet_NaN(), "test context");
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("NaN") != std::string::npos);
        assert(msg.find("test context") != std::string::npos);
        assert(msg.find("numerical instability") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ validateFinite detects NaN with helpful message" << std::endl;

    // Test validateFinite with Inf
    caught = false;
    try {
        Validation::validateFinite(std::numeric_limits<double>::infinity(), "test context");
    } catch (const std::runtime_error& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("Inf") != std::string::npos);
        assert(msg.find("test context") != std::string::npos);
        assert(msg.find("overflow") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ validateFinite detects Inf with helpful message" << std::endl;

    // Test validatePositive
    caught = false;
    try {
        Validation::validatePositive(-1.0, "learning_rate");
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("learning_rate") != std::string::npos);
        assert(msg.find("positive") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ validatePositive detects negative values" << std::endl;

    // Test validatePositiveSize
    caught = false;
    try {
        Validation::validatePositiveSize(0, "layer_size");
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();
        assert(msg.find("layer_size") != std::string::npos);
        assert(msg.find("greater than zero") != std::string::npos);
    }
    assert(caught);
    std::cout << "  ✓ validatePositiveSize detects zero values" << std::endl;

    std::cout << "  PASSED: Validation utilities" << std::endl;
}

// Test 8: Error message informativeness
void testErrorMessageQuality() {
    std::cout << "\nTest 8: Error message informativeness..." << std::endl;

    // All error messages should include:
    // 1. What operation failed
    // 2. What was expected
    // 3. What was received
    // 4. Helpful context or suggestions

    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());

    bool caught = false;
    try {
        Vector wrong_input = {1.0, 2.0, 3.0};
        net.predict(wrong_input);
    } catch (const std::invalid_argument& e) {
        caught = true;
        std::string msg = e.what();

        // Check for operation
        assert(msg.find("predict") != std::string::npos);

        // Check for expected value
        assert(msg.find("2") != std::string::npos);

        // Check for received value
        assert(msg.find("3") != std::string::npos);

        // Check for context
        assert(msg.find("input size") != std::string::npos);

        std::cout << "  Example error message: " << msg << std::endl;
    }
    assert(caught);
    std::cout << "  ✓ Error messages include operation, expected, received, and context" << std::endl;

    std::cout << "  PASSED: Error message quality" << std::endl;
}

int main() {
    std::cout << "=== Error Handling and Validation Tests (Task 15.1) ===" << std::endl;
    std::cout << "Testing Requirements 12.1-12.5" << std::endl;
    std::cout << std::endl;

    try {
        testConfigurationValidation();
        testDimensionMismatchMessages();
        testNaNDetectionForwardPass();
        testInfDetectionForwardPass();
        testNaNDetectionLoss();
        testNaNDetectionBackprop();
        testValidationUtilities();
        testErrorMessageQuality();

        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n=== TEST FAILED ===" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
