/**
 * Property-based tests for error handling (Task 15.3)
 *
 * Tests Properties 26, 32, 33:
 * - Property 26: Dimension Mismatch Detection
 * - Property 32: Invalid Configuration Rejection
 * - Property 33: Error Message Informativeness
 *
 * Validates Requirements 9.6, 12.2, 12.3, 12.5
 */

#include "network.h"
#include "layer.h"
#include "activation.h"
#include "loss.h"
#include "matrix.h"
#include "vector.h"
#include "validation.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <iostream>
#include <stdexcept>
#include <string>

// Helper function to check if exception message contains expected text
bool messageContains(const std::exception& e, const std::string& expected) {
    std::string msg = e.what();
    return msg.find(expected) != std::string::npos;
}

// Custom generator for positive sizes
rc::Gen<size_t> genPositiveSize() {
    return rc::gen::inRange<size_t>(1, 20);
}

// Custom generator for invalid (zero or negative) learning rates
rc::Gen<double> genInvalidLearningRate() {
    return rc::gen::element(
        0.0,
        -0.001,
        -0.1,
        -1.0,
        -10.0
    );
}

// Custom generator for valid learning rates
rc::Gen<double> genValidLearningRate() {
    return rc::gen::map(rc::gen::inRange(1, 1000), [](int i) {
        return i / 1000.0;  // Generates values from 0.001 to 1.0
    });
}

// ============================================================================
// PROPERTY 26: DIMENSION MISMATCH DETECTION
// ============================================================================

// **Validates: Requirements 9.6, 12.3**
// Feature: neural-network-framework, Property 26: Dimension Mismatch Detection
// For any matrix or vector operation where operand dimensions are incompatible,
// the operation should be rejected with a descriptive error indicating expected vs actual dimensions.
RC_GTEST_PROP(ErrorHandlingPropertyTest, MatrixAdditionDimensionMismatch, ()) {
    // Generate two matrices with different dimensions
    auto rows1 = *genPositiveSize();
    auto cols1 = *genPositiveSize();
    auto rows2 = *genPositiveSize();
    auto cols2 = *genPositiveSize();

    // Ensure dimensions are different
    RC_PRE(rows1 != rows2 || cols1 != cols2);

    Matrix m1(rows1, cols1);
    Matrix m2(rows2, cols2);

    // Attempt matrix addition - should throw
    bool caught = false;
    try {
        Matrix result = m1 + m2;
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught = true;
        // Verify error message contains context
        RC_ASSERT(messageContains(e, "addition"));
        RC_ASSERT(messageContains(e, "dimensions mismatch") || messageContains(e, "dimension"));
    }

    RC_ASSERT(caught);
}

// **Validates: Requirements 9.6, 12.3**
// Feature: neural-network-framework, Property 26: Dimension Mismatch Detection
// Matrix multiplication with incompatible dimensions should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, MatrixMultiplicationDimensionMismatch, ()) {
    // Generate two matrices with incompatible dimensions for multiplication
    auto m = *genPositiveSize();
    auto n = *genPositiveSize();
    auto p = *genPositiveSize();

    // Ensure n != p so A(m×n) * B(p×q) is invalid
    RC_PRE(n != p);

    auto q = *genPositiveSize();

    Matrix A(m, n);
    Matrix B(p, q);

    // Attempt matrix multiplication - should throw
    bool caught = false;
    try {
        Matrix result = A * B;
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught = true;
        // Verify error message contains context
        RC_ASSERT(messageContains(e, "multiplication") || messageContains(e, "multiply"));
        RC_ASSERT(messageContains(e, "incompatible") || messageContains(e, "dimension"));
    }

    RC_ASSERT(caught);
}

// **Validates: Requirements 9.6, 12.3**
// Feature: neural-network-framework, Property 26: Dimension Mismatch Detection
// Matrix-vector multiplication with incompatible dimensions should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, MatrixVectorMultiplicationDimensionMismatch, ()) {
    // Generate matrix and vector with incompatible dimensions
    auto rows = *genPositiveSize();
    auto cols = *genPositiveSize();
    auto vec_size = *genPositiveSize();

    // Ensure cols != vec_size so M(rows×cols) * v(vec_size) is invalid
    RC_PRE(cols != vec_size);

    Matrix m(rows, cols);
    Vector v(vec_size);

    // Attempt matrix-vector multiplication - should throw
    bool caught = false;
    try {
        Vector result = m * v;
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught = true;
        // Verify error message contains context
        RC_ASSERT(messageContains(e, "multiplication") || messageContains(e, "multiply"));
        RC_ASSERT(messageContains(e, "incompatible") || messageContains(e, "dimension"));
    }

    RC_ASSERT(caught);
}

// **Validates: Requirements 9.6, 12.3**
// Feature: neural-network-framework, Property 26: Dimension Mismatch Detection
// Vector operations with incompatible dimensions should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, VectorOperationsDimensionMismatch, ()) {
    // Generate two vectors with different sizes
    auto size1 = *genPositiveSize();
    auto size2 = *genPositiveSize();

    // Ensure sizes are different
    RC_PRE(size1 != size2);

    Vector v1(size1);
    Vector v2(size2);

    // Test addition
    bool caught_add = false;
    try {
        Vector result = v1 + v2;
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught_add = true;
        RC_ASSERT(messageContains(e, "addition") || messageContains(e, "add"));
        RC_ASSERT(messageContains(e, "dimensions mismatch") || messageContains(e, "dimension"));
    }
    RC_ASSERT(caught_add);

    // Test subtraction
    bool caught_sub = false;
    try {
        Vector result = v1 - v2;
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught_sub = true;
        RC_ASSERT(messageContains(e, "subtraction") || messageContains(e, "subtract"));
        RC_ASSERT(messageContains(e, "dimensions mismatch") || messageContains(e, "dimension"));
    }
    RC_ASSERT(caught_sub);

    // Test dot product
    bool caught_dot = false;
    try {
        double result = v1.dot(v2);
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught_dot = true;
        RC_ASSERT(messageContains(e, "dot product") || messageContains(e, "dot"));
        RC_ASSERT(messageContains(e, "dimensions mismatch") || messageContains(e, "dimension"));
    }
    RC_ASSERT(caught_dot);
}

// **Validates: Requirements 9.6, 12.3**
// Feature: neural-network-framework, Property 26: Dimension Mismatch Detection
// Network input dimension mismatch should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, NetworkInputDimensionMismatch, ()) {
    // Generate network with specific input size
    auto input_size = *genPositiveSize();
    auto hidden_size = *genPositiveSize();
    auto output_size = *genPositiveSize();

    Network net;
    net.addLayer(input_size, hidden_size, std::make_shared<Sigmoid>());
    net.addLayer(hidden_size, output_size, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(input_size, hidden_size);
    net.getLayer(1).initializeXavier(hidden_size, output_size);

    // Generate input with wrong size
    auto wrong_input_size = *genPositiveSize();
    RC_PRE(wrong_input_size != input_size);

    Vector wrong_input(wrong_input_size);

    // Attempt prediction with wrong input size - should throw
    bool caught = false;
    try {
        Vector result = net.predict(wrong_input);
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught = true;
        // Verify error message contains context
        RC_ASSERT(messageContains(e, "input size") || messageContains(e, "input"));
        RC_ASSERT(messageContains(e, "does not match") || messageContains(e, "mismatch") || messageContains(e, "expected"));
    }

    RC_ASSERT(caught);
}

// **Validates: Requirements 9.6, 12.3**
// Feature: neural-network-framework, Property 26: Dimension Mismatch Detection
// Loss function dimension mismatch should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, LossFunctionDimensionMismatch, ()) {
    // Generate vectors with different sizes
    auto size1 = *genPositiveSize();
    auto size2 = *genPositiveSize();

    // Ensure sizes are different
    RC_PRE(size1 != size2);

    Vector predicted(size1);
    Vector target(size2);

    // Test MSE
    bool caught_mse = false;
    try {
        MeanSquaredError mse;
        double loss = mse.compute(predicted, target);
        (void)loss;
    } catch (const std::invalid_argument& e) {
        caught_mse = true;
        RC_ASSERT(messageContains(e, "MSE") || messageContains(e, "predicted") || messageContains(e, "target"));
        RC_ASSERT(messageContains(e, "size") || messageContains(e, "dimension"));
    }
    RC_ASSERT(caught_mse);

    // Test CrossEntropy
    bool caught_ce = false;
    try {
        CrossEntropy ce;
        double loss = ce.compute(predicted, target);
        (void)loss;
    } catch (const std::invalid_argument& e) {
        caught_ce = true;
        RC_ASSERT(messageContains(e, "CrossEntropy") || messageContains(e, "predicted") || messageContains(e, "target"));
        RC_ASSERT(messageContains(e, "size") || messageContains(e, "dimension"));
    }
    RC_ASSERT(caught_ce);
}

// ============================================================================
// PROPERTY 32: INVALID CONFIGURATION REJECTION
// ============================================================================

// **Validates: Requirements 12.2**
// Feature: neural-network-framework, Property 32: Invalid Configuration Rejection
// For any training configuration with invalid parameters (negative learning rate, zero epochs,
// negative layer sizes), the system should reject the configuration before starting training
// and provide a descriptive error.
RC_GTEST_PROP(ErrorHandlingPropertyTest, InvalidLearningRateRejection, ()) {
    // Generate invalid learning rate (zero or negative)
    auto invalid_lr = *genInvalidLearningRate();

    // Create a simple network
    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);
    net.getLayer(1).initializeXavier(3, 1);

    std::vector<Vector> inputs = {{0.0, 0.0}};
    std::vector<Vector> targets = {{0.0}};
    MeanSquaredError mse;

    // Attempt training with invalid learning rate - should throw
    bool caught = false;
    try {
        net.train(inputs, targets, 1, invalid_lr, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        // Verify error message contains context
        RC_ASSERT(messageContains(e, "learning_rate") || messageContains(e, "learning rate"));
        RC_ASSERT(messageContains(e, "positive") || messageContains(e, "greater than zero"));
    }

    RC_ASSERT(caught);
}

// **Validates: Requirements 12.2**
// Feature: neural-network-framework, Property 32: Invalid Configuration Rejection
// Zero epochs should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, ZeroEpochsRejection, ()) {
    // Create a simple network
    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);
    net.getLayer(1).initializeXavier(3, 1);

    std::vector<Vector> inputs = {{0.0, 0.0}};
    std::vector<Vector> targets = {{0.0}};
    MeanSquaredError mse;

    auto valid_lr = *genValidLearningRate();

    // Attempt training with zero epochs - should throw
    bool caught = false;
    try {
        net.train(inputs, targets, 0, valid_lr, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        // Verify error message contains context
        RC_ASSERT(messageContains(e, "epochs"));
        RC_ASSERT(messageContains(e, "greater than zero") || messageContains(e, "positive"));
    }

    RC_ASSERT(caught);
}

// **Validates: Requirements 12.2**
// Feature: neural-network-framework, Property 32: Invalid Configuration Rejection
// Invalid batch size should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, InvalidBatchSizeRejection, ()) {
    // Create a simple network
    Network net;
    net.addLayer(2, 3, std::make_shared<Sigmoid>());
    net.addLayer(3, 1, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 3);
    net.getLayer(1).initializeXavier(3, 1);

    // Generate dataset
    auto dataset_size = *rc::gen::inRange<size_t>(1, 10);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    for (size_t i = 0; i < dataset_size; ++i) {
        inputs.push_back(Vector(2));
        targets.push_back(Vector(1));
    }

    MeanSquaredError mse;
    auto valid_lr = *genValidLearningRate();

    // Test zero batch size
    bool caught_zero = false;
    try {
        net.train(inputs, targets, 1, valid_lr, mse, 0);
    } catch (const std::invalid_argument& e) {
        caught_zero = true;
        RC_ASSERT(messageContains(e, "batch_size") || messageContains(e, "batch size"));
        RC_ASSERT(messageContains(e, "greater than zero") || messageContains(e, "positive"));
    }
    RC_ASSERT(caught_zero);

    // Test batch size larger than dataset
    auto invalid_batch_size = dataset_size + *rc::gen::inRange<size_t>(1, 10);
    bool caught_large = false;
    try {
        net.train(inputs, targets, 1, valid_lr, mse, invalid_batch_size);
    } catch (const std::invalid_argument& e) {
        caught_large = true;
        RC_ASSERT(messageContains(e, "batch_size") || messageContains(e, "batch size"));
        RC_ASSERT(messageContains(e, "dataset size") || messageContains(e, "dataset"));
    }
    RC_ASSERT(caught_large);
}

// **Validates: Requirements 12.2**
// Feature: neural-network-framework, Property 32: Invalid Configuration Rejection
// Invalid layer sizes should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, InvalidLayerSizesRejection, ()) {
    Network net;

    auto valid_size = *genPositiveSize();

    // Test zero input size
    bool caught_input = false;
    try {
        net.addLayer(0, valid_size, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught_input = true;
        RC_ASSERT(messageContains(e, "input_size") || messageContains(e, "input size"));
        RC_ASSERT(messageContains(e, "greater than zero") || messageContains(e, "positive"));
    }
    RC_ASSERT(caught_input);

    // Test zero output size
    bool caught_output = false;
    try {
        Network net2;
        net2.addLayer(valid_size, 0, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught_output = true;
        RC_ASSERT(messageContains(e, "output_size") || messageContains(e, "output size"));
        RC_ASSERT(messageContains(e, "greater than zero") || messageContains(e, "positive"));
    }
    RC_ASSERT(caught_output);
}

// **Validates: Requirements 12.2**
// Feature: neural-network-framework, Property 32: Invalid Configuration Rejection
// Layer connectivity mismatch should be rejected
RC_GTEST_PROP(ErrorHandlingPropertyTest, LayerConnectivityMismatchRejection, ()) {
    // Generate layer sizes
    auto input_size = *genPositiveSize();
    auto hidden_size = *genPositiveSize();
    auto wrong_size = *genPositiveSize();

    // Ensure wrong_size != hidden_size
    RC_PRE(wrong_size != hidden_size);

    Network net;
    net.addLayer(input_size, hidden_size, std::make_shared<Sigmoid>());

    // Attempt to add layer with mismatched input size - should throw
    bool caught = false;
    try {
        net.addLayer(wrong_size, 1, std::make_shared<Sigmoid>());
    } catch (const std::invalid_argument& e) {
        caught = true;
        RC_ASSERT(messageContains(e, "layer connectivity") || messageContains(e, "input size"));
        RC_ASSERT(messageContains(e, "validation failed") ||
                  messageContains(e, "must match") ||
                  messageContains(e, "does not match") ||
                  messageContains(e, "mismatch"));
    }

    RC_ASSERT(caught);
}

// ============================================================================
// PROPERTY 33: ERROR MESSAGE INFORMATIVENESS
// ============================================================================

// **Validates: Requirements 12.5**
// Feature: neural-network-framework, Property 33: Error Message Informativeness
// For any error condition (dimension mismatch, invalid input, corrupted file),
// the error message should include context about what operation failed and
// what was expected vs what was received.
RC_GTEST_PROP(ErrorHandlingPropertyTest, ErrorMessagesContainContext, ()) {
    // Generate incompatible matrix dimensions
    auto rows1 = *genPositiveSize();
    auto cols1 = *genPositiveSize();
    auto rows2 = *genPositiveSize();
    auto cols2 = *genPositiveSize();

    RC_PRE(rows1 != rows2 || cols1 != cols2);

    Matrix m1(rows1, cols1);
    Matrix m2(rows2, cols2);

    // Attempt operation and verify error message contains:
    // 1. Operation name
    // 2. Expected dimensions
    // 3. Actual dimensions
    bool caught = false;
    std::string error_msg;
    try {
        Matrix result = m1 + m2;
        (void)result;
    } catch (const std::invalid_argument& e) {
        caught = true;
        error_msg = e.what();

        // Check for operation name
        RC_ASSERT(messageContains(e, "addition") || messageContains(e, "add"));

        // Check for dimension information
        // Error message should mention dimensions or sizes
        RC_ASSERT(messageContains(e, "dimension") || messageContains(e, "size"));

        // Check for mismatch indication
        RC_ASSERT(messageContains(e, "mismatch") ||
                  messageContains(e, "incompatible") ||
                  messageContains(e, "does not match"));
    }

    RC_ASSERT(caught);
    RC_ASSERT(!error_msg.empty());
}

// **Validates: Requirements 12.5**
// Feature: neural-network-framework, Property 33: Error Message Informativeness
// Network input size error messages should be informative
RC_GTEST_PROP(ErrorHandlingPropertyTest, NetworkInputErrorMessageInformative, ()) {
    auto expected_size = *genPositiveSize();
    auto actual_size = *genPositiveSize();

    RC_PRE(expected_size != actual_size);

    Network net;
    net.addLayer(expected_size, 2, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(expected_size, 2);

    Vector wrong_input(actual_size);

    bool caught = false;
    std::string error_msg;
    try {
        net.predict(wrong_input);
    } catch (const std::invalid_argument& e) {
        caught = true;
        error_msg = e.what();

        // Error message should mention:
        // 1. That it's an input size issue
        RC_ASSERT(messageContains(e, "input") || messageContains(e, "Input"));

        // 2. Size or dimension information
        RC_ASSERT(messageContains(e, "size") || messageContains(e, "dimension"));

        // 3. Mismatch or expectation
        RC_ASSERT(messageContains(e, "does not match") ||
                  messageContains(e, "mismatch") ||
                  messageContains(e, "expected"));
    }

    RC_ASSERT(caught);
    RC_ASSERT(!error_msg.empty());
}

// **Validates: Requirements 12.5**
// Feature: neural-network-framework, Property 33: Error Message Informativeness
// Configuration error messages should be informative
RC_GTEST_PROP(ErrorHandlingPropertyTest, ConfigurationErrorMessageInformative, ()) {
    auto invalid_lr = *genInvalidLearningRate();

    Network net;
    net.addLayer(2, 2, std::make_shared<Sigmoid>());
    net.getLayer(0).initializeXavier(2, 2);

    std::vector<Vector> inputs = {{0.0, 0.0}};
    std::vector<Vector> targets = {{0.0, 0.0}};
    MeanSquaredError mse;

    bool caught = false;
    std::string error_msg;
    try {
        net.train(inputs, targets, 1, invalid_lr, mse);
    } catch (const std::invalid_argument& e) {
        caught = true;
        error_msg = e.what();

        // Error message should mention:
        // 1. The parameter name
        RC_ASSERT(messageContains(e, "learning_rate") || messageContains(e, "learning rate"));

        // 2. The constraint
        RC_ASSERT(messageContains(e, "positive") ||
                  messageContains(e, "greater than zero") ||
                  messageContains(e, "must be"));
    }

    RC_ASSERT(caught);
    RC_ASSERT(!error_msg.empty());
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
