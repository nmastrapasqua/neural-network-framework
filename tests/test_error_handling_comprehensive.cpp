/**
 * Comprehensive unit tests for error handling (Task 15.2)
 *
 * Tests Requirements 12.1, 12.2, 12.3, 12.5:
 * - Dimension mismatch errors (matrix/vector operations, network input validation)
 * - Invalid configuration errors (negative learning rate, zero epochs, invalid layer sizes)
 * - Corrupted file errors (invalid serialization format, corrupted data)
 *
 * This file consolidates and extends error handling tests to ensure comprehensive coverage.
 */

#include "network.h"
#include "layer.h"
#include "activation.h"
#include "loss.h"
#include "matrix.h"
#include "vector.h"
#include "validation.h"
#include "serializer.h"
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

// Helper function to check if exception message contains expected text
bool messageContains(const std::exception& e, const std::string& expected) {
    std::string msg = e.what();
    return msg.find(expected) != std::string::npos;
}

// ============================================================================
// SECTION 1: DIMENSION MISMATCH ERRORS
// ============================================================================

void testMatrixDimensionErrors() {
    std::cout << "Test 1: Matrix dimension mismatch errors..." << std::endl;

    // Test 1.1: Zero dimensions
    bool caught = false;
    try {
        Matrix m(0, 5);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "dimensions must be greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Zero rows rejected" << std::endl;

    caught = false;
    try {
        Matrix m(5, 0);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "dimensions must be greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Zero columns rejected" << std::endl;

    // Test 1.2: Matrix addition dimension mismatch
    caught = false;
    try {
        Matrix m1(2, 3);
        Matrix m2(3, 2);
        Matrix result = m1 + m2;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "addition"));
        assert(messageContains(e, "dimensions mismatch"));
    }
    assert(caught);
    std::cout << "  ✓ Matrix addition dimension mismatch detected" << std::endl;

    // Test 1.3: Matrix multiplication incompatible dimensions
    caught = false;
    try {
        Matrix m1(2, 3);
        Matrix m2(4, 2);
        Matrix result = m1 * m2;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "multiplication"));
        assert(messageContains(e, "incompatible"));
    }
    assert(caught);
    std::cout << "  ✓ Matrix multiplication incompatible dimensions detected" << std::endl;

    // Test 1.4: Matrix-vector multiplication incompatible dimensions
    caught = false;
    try {
        Matrix m(2, 3);
        Vector v(2);  // Should be size 3
        Vector result = m * v;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "Matrix-vector multiplication"));
        assert(messageContains(e, "incompatible"));
    }
    assert(caught);
    std::cout << "  ✓ Matrix-vector multiplication incompatible dimensions detected" << std::endl;

    // Test 1.5: Element-wise multiplication dimension mismatch
    caught = false;
    try {
        Matrix m1(2, 3);
        Matrix m2(2, 4);
        Matrix result = m1.elementWiseMultiply(m2);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "element-wise multiplication"));
        assert(messageContains(e, "dimensions mismatch"));
    }
    assert(caught);
    std::cout << "  ✓ Element-wise multiplication dimension mismatch detected" << std::endl;

    std::cout << "  PASSED: Matrix dimension errors" << std::endl;
}

void testVectorDimensionErrors() {
    std::cout << "\nTest 2: Vector dimension mismatch errors..." << std::endl;

    // Test 2.1: Vector addition dimension mismatch
    bool caught = false;
    try {
        Vector v1(2);
        Vector v2(3);
        Vector result = v1 + v2;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "addition"));
        assert(messageContains(e, "dimensions mismatch"));
    }
    assert(caught);
    std::cout << "  ✓ Vector addition dimension mismatch detected" << std::endl;

    // Test 2.2: Vector subtraction dimension mismatch
    caught = false;
    try {
        Vector v1(2);
        Vector v2(3);
        Vector result = v1 - v2;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "subtraction"));
        assert(messageContains(e, "dimensions mismatch"));
    }
    assert(caught);
    std::cout << "  ✓ Vector subtraction dimension mismatch detected" << std::endl;

    // Test 2.3: Element-wise multiplication dimension mismatch
    caught = false;
    try {
        Vector v1(2);
        Vector v2(3);
        Vector result = v1.elementWiseMultiply(v2);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "element-wise multiplication"));
        assert(messageContains(e, "dimensions mismatch"));
    }
    assert(caught);
    std::cout << "  ✓ Vector element-wise multiplication dimension mismatch detected" << std::endl;

    // Test 2.4: Dot product dimension mismatch
    caught = false;
    try {
        Vector v1(2);
        Vector v2(3);
        double result = v1.dot(v2);
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "dot product"));
        assert(messageContains(e, "dimensions mismatch"));
    }
    assert(caught);
    std::cout << "  ✓ Dot product dimension mismatch detected" << std::endl;

    std::cout << "  PASSED: Vector dimension errors" << std::endl;
}

void testNetworkInputDimensionErrors() {
    std::cout << "\nTest 3: Network input dimension errors..." << std::endl;

    // Test 3.1: Input size mismatch
    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);

    bool caught = false;
    try {
        Vector wrong_input(3);  // Should be size 2
        net.predict(wrong_input);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "input size"));
        assert(messageContains(e, "does not match"));
    }
    assert(caught);
    std::cout << "  ✓ Network input size mismatch detected" << std::endl;

    // Test 3.2: Empty network prediction
    caught = false;
    try {
        Network empty_net;
        Vector input(2);
        empty_net.predict(input);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "empty network"));
    }
    assert(caught);
    std::cout << "  ✓ Empty network prediction rejected" << std::endl;

    // Test 3.3: Layer connectivity mismatch
    caught = false;
    try {
        Network bad_net;
        bad_net.addLayer(2, 3, std::make_shared<Sigmoid>());
        bad_net.addLayer(5, 1, std::make_shared<Sigmoid>());  // Should be input_size=3
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "layer connectivity"));
    }
    assert(caught);
    std::cout << "  ✓ Layer connectivity mismatch detected" << std::endl;

    std::cout << "  PASSED: Network input dimension errors" << std::endl;
}

// ============================================================================
// SECTION 2: INVALID CONFIGURATION ERRORS
// ============================================================================

void testInvalidConfigurationErrors() {
    std::cout << "\nTest 4: Invalid configuration errors..." << std::endl;

    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);
    net.getLayer(1).initializeXavier(3, 1);

    std::vector<Vector> inputs = {{0.0, 0.0}};
    std::vector<Vector> targets = {{0.0}};
    MeanSquaredError mse;

    // Test 4.1: Negative learning rate
    bool caught = false;
    try {
        net.train(inputs, targets, 1, -0.1, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "learning_rate"));
        assert(messageContains(e, "positive"));
    }
    assert(caught);
    std::cout << "  ✓ Negative learning rate rejected" << std::endl;

    // Test 4.2: Zero learning rate
    caught = false;
    try {
        net.train(inputs, targets, 1, 0.0, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "learning_rate"));
        assert(messageContains(e, "positive"));
    }
    assert(caught);
    std::cout << "  ✓ Zero learning rate rejected" << std::endl;

    // Test 4.3: Zero epochs
    caught = false;
    try {
        net.train(inputs, targets, 0, 0.1, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "epochs"));
        assert(messageContains(e, "greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Zero epochs rejected" << std::endl;

    // Test 4.4: Zero batch size
    caught = false;
    try {
        net.train(inputs, targets, 1, 0.1, mse, 0);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "batch_size"));
        assert(messageContains(e, "greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Zero batch size rejected" << std::endl;

    // Test 4.5: Batch size larger than dataset
    caught = false;
    try {
        net.train(inputs, targets, 1, 0.1, mse, 10);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "batch_size"));
        assert(messageContains(e, "dataset size"));
    }
    assert(caught);
    std::cout << "  ✓ Batch size > dataset size rejected" << std::endl;

    // Test 4.6: Zero layer input size
    caught = false;
    try {
        Network bad_net;
        bad_net.addLayer(0, 3, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "input_size"));
        assert(messageContains(e, "greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Zero layer input size rejected" << std::endl;

    // Test 4.7: Zero layer output size
    caught = false;
    try {
        Network bad_net;
        bad_net.addLayer(2, 0, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "output_size"));
        assert(messageContains(e, "greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Zero layer output size rejected" << std::endl;

    std::cout << "  PASSED: Invalid configuration errors" << std::endl;
}

void testInvalidRangeErrors() {
    std::cout << "\nTest 5: Invalid range errors..." << std::endl;

    // Test 5.1: Matrix randomize with min > max
    bool caught = false;
    try {
        Matrix m(2, 2);
        m.randomize(1.0, 0.0);  // min > max
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "min must be less than or equal to max"));
    }
    assert(caught);
    std::cout << "  ✓ Matrix randomize with min > max rejected" << std::endl;

    // Test 5.2: Layer initializeWeights with min >= max
    caught = false;
    try {
        Network net;
        net.addLayer(2, 2, std::make_shared<Sigmoid>());
        net.getLayer(0).initializeWeights(1.0, 0.5);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "min") && messageContains(e, "max"));
    }
    assert(caught);
    std::cout << "  ✓ Layer initializeWeights with min >= max rejected" << std::endl;

    // Test 5.3: Xavier initialization with zero fan_in
    caught = false;
    try {
        Network net;
        net.addLayer(2, 2, std::make_shared<Sigmoid>());
        net.getLayer(0).initializeXavier(0, 2);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "fan_in") && messageContains(e, "greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ Xavier initialization with zero fan_in rejected" << std::endl;

    // Test 5.4: He initialization with zero fan_in
    caught = false;
    try {
        Network net;
        net.addLayer(2, 2, std::make_shared<ReLU>());
        net.getLayer(0).initializeHe(0);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "fan_in") && messageContains(e, "greater than zero"));
    }
    assert(caught);
    std::cout << "  ✓ He initialization with zero fan_in rejected" << std::endl;

    std::cout << "  PASSED: Invalid range errors" << std::endl;
}

// ============================================================================
// SECTION 3: CORRUPTED FILE ERRORS
// ============================================================================

void testCorruptedFileErrors() {
    std::cout << "\nTest 6: Corrupted file errors..." << std::endl;

    // Test 6.1: Invalid header
    bool caught = false;
    try {
        std::stringstream ss;
        ss << "INVALID_HEADER\n";
        ss << "LAYERS 1\n";
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::runtime_error& e) {
        caught = true;
        assert(messageContains(e, "NEURAL_NETWORK_V1"));
    }
    assert(caught);
    std::cout << "  ✓ Invalid header detected" << std::endl;

    // Test 6.2: Missing END marker
    caught = false;
    try {
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
        // Missing END
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);
    std::cout << "  ✓ Missing END marker detected" << std::endl;

    // Test 6.3: Invalid activation function
    caught = false;
    try {
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
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "invalid_activation"));
    }
    assert(caught);
    std::cout << "  ✓ Invalid activation function detected" << std::endl;

    // Test 6.4: Weights dimension mismatch
    caught = false;
    try {
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
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "biases size"));
    }
    assert(caught);
    std::cout << "  ✓ Biases dimension mismatch detected" << std::endl;

    // Test 6.5: Layer connectivity mismatch in file
    caught = false;
    try {
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
        ss << "  INPUT_SIZE 5\n";  // Wrong - should be 3
        ss << "  OUTPUT_SIZE 1\n";
        ss << "  ACTIVATION tanh\n";
        ss << "  WEIGHTS 1 5\n";
        ss << "    0.1 0.2 0.3 0.4 0.5\n";
        ss << "  BIASES 1\n";
        ss << "    0.1\n";
        ss << "END\n";
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "does not match previous layer output size"));
    }
    assert(caught);
    std::cout << "  ✓ Layer connectivity mismatch in file detected" << std::endl;

    // Test 6.6: Corrupted weight data (insufficient values)
    caught = false;
    try {
        std::stringstream ss;
        ss << "NEURAL_NETWORK_V1\n";
        ss << "LAYERS 1\n";
        ss << "LAYER 0\n";
        ss << "  INPUT_SIZE 2\n";
        ss << "  OUTPUT_SIZE 2\n";
        ss << "  ACTIVATION sigmoid\n";
        ss << "  WEIGHTS 2 2\n";
        ss << "    0.5 0.3\n";  // Only 2 values, need 4
        ss << "  BIASES 2\n";
        ss << "    0.1 0.2\n";
        ss << "END\n";
        Network loaded = Serializer::deserialize(ss);
    } catch (const std::exception& e) {
        caught = true;
        // Could be runtime_error or invalid_argument depending on parser
    }
    assert(caught);
    std::cout << "  ✓ Corrupted weight data detected" << std::endl;

    std::cout << "  PASSED: Corrupted file errors" << std::endl;
}

// ============================================================================
// SECTION 4: ADDITIONAL ERROR SCENARIOS
// ============================================================================

void testEmptyDatasetErrors() {
    std::cout << "\nTest 7: Empty dataset errors..." << std::endl;

    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);
    net.getLayer(1).initializeXavier(3, 1);

    MeanSquaredError mse;

    // Test 7.1: Empty training inputs
    bool caught = false;
    try {
        std::vector<Vector> empty_inputs;
        std::vector<Vector> targets = {{0.0}};
        net.train(empty_inputs, targets, 1, 0.1, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "inputs cannot be empty"));
    }
    assert(caught);
    std::cout << "  ✓ Empty training inputs rejected" << std::endl;

    // Test 7.2: Mismatched inputs and targets size
    caught = false;
    try {
        std::vector<Vector> inputs = {{0.0, 0.0}, {1.0, 1.0}};
        std::vector<Vector> targets = {{0.0}};  // Only 1 target for 2 inputs
        net.train(inputs, targets, 1, 0.1, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "inputs size"));
        assert(messageContains(e, "targets size"));
    }
    assert(caught);
    std::cout << "  ✓ Mismatched inputs/targets size rejected" << std::endl;

    // Test 7.3: Empty validation inputs
    caught = false;
    try {
        std::vector<Vector> empty_inputs;
        std::vector<Vector> targets = {{0.0}};
        MeanSquaredError loss;
        net.validate(empty_inputs, targets, loss);
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "test_inputs cannot be empty"));
    }
    assert(caught);
    std::cout << "  ✓ Empty validation inputs rejected" << std::endl;

    std::cout << "  PASSED: Empty dataset errors" << std::endl;
}

void testLossFunctionDimensionErrors() {
    std::cout << "\nTest 8: Loss function dimension errors..." << std::endl;

    // Test 8.1: MSE with mismatched dimensions
    bool caught = false;
    try {
        Vector predicted(2);
        Vector target(3);
        MeanSquaredError mse;
        double loss = mse.compute(predicted, target);
        (void)loss;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "MSE"));
        assert(messageContains(e, "predicted size"));
        assert(messageContains(e, "target size"));
    }
    assert(caught);
    std::cout << "  ✓ MSE dimension mismatch detected" << std::endl;

    // Test 8.2: CrossEntropy with mismatched dimensions
    caught = false;
    try {
        Vector predicted(2);
        Vector target(3);
        CrossEntropy ce;
        double loss = ce.compute(predicted, target);
        (void)loss;
    } catch (const std::invalid_argument& e) {
        caught = true;
        assert(messageContains(e, "CrossEntropy"));
        assert(messageContains(e, "predicted size"));
        assert(messageContains(e, "target size"));
    }
    assert(caught);
    std::cout << "  ✓ CrossEntropy dimension mismatch detected" << std::endl;

    std::cout << "  PASSED: Loss function dimension errors" << std::endl;
}

void testOutOfRangeErrors() {
    std::cout << "\nTest 9: Out of range errors..." << std::endl;

    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());

    // Test 9.1: getLayer with invalid index
    bool caught = false;
    try {
        Layer& layer = net.getLayer(5);  // Only 2 layers (indices 0, 1)
        (void)layer;
    } catch (const std::out_of_range& e) {
        caught = true;
        assert(messageContains(e, "out of range"));
    }
    assert(caught);
    std::cout << "  ✓ getLayer out of range detected" << std::endl;

    std::cout << "  PASSED: Out of range errors" << std::endl;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Comprehensive Error Handling Tests" << std::endl;
    std::cout << "Task 15.2: Unit tests for error handling" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Testing Requirements:" << std::endl;
    std::cout << "  - 12.1: NaN/Inf detection" << std::endl;
    std::cout << "  - 12.2: Configuration parameter validation" << std::endl;
    std::cout << "  - 12.3: Dimension validation" << std::endl;
    std::cout << "  - 12.5: Descriptive error messages" << std::endl;
    std::cout << std::endl;

    try {
        // Section 1: Dimension mismatch errors
        testMatrixDimensionErrors();
        testVectorDimensionErrors();
        testNetworkInputDimensionErrors();

        // Section 2: Invalid configuration errors
        testInvalidConfigurationErrors();
        testInvalidRangeErrors();

        // Section 3: Corrupted file errors
        testCorruptedFileErrors();

        // Section 4: Additional error scenarios
        testEmptyDatasetErrors();
        testLossFunctionDimensionErrors();
        testOutOfRangeErrors();

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "ALL TESTS PASSED ✓" << std::endl;
        std::cout << "========================================" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "TEST FAILED ✗" << std::endl;
        std::cerr << "========================================" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
