#include "activation.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <limits>
#include <algorithm>

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
rc::Gen<std::shared_ptr<ActivationFunction>> arbActivationFunction() {
    return rc::gen::map(
        rc::gen::inRange(0, 3),
        [](int choice) -> std::shared_ptr<ActivationFunction> {
            switch (choice) {
                case 0: return std::make_shared<Sigmoid>();
                case 1: return std::make_shared<Tanh>();
                case 2: return std::make_shared<ReLU>();
                default: return std::make_shared<Sigmoid>();
            }
        }
    );
}

// **Validates: Requirements 2.1, 2.2, 2.3, 2.6**
// Feature: neural-network-framework, Property 5: Activation Function Mathematical Correctness
// Verify that Sigmoid, Tanh, and ReLU produce mathematically correct outputs according to their definitions
RC_GTEST_PROP(ActivationPropertyTest, ActivationFunctionMathematicalCorrectness, ()) {
    // Generate random input value in a reasonable range
    auto x = *genDoubleInRange(-10.0, 10.0);

    // Test Sigmoid: f(x) = 1 / (1 + e^(-x))
    Sigmoid sigmoid;
    double sigmoid_output = sigmoid.activate(x);
    double expected_sigmoid = 1.0 / (1.0 + std::exp(-x));

    // Sigmoid output should be in range (0, 1)
    RC_ASSERT(sigmoid_output > 0.0);
    RC_ASSERT(sigmoid_output < 1.0);
    RC_ASSERT(approxEqual(sigmoid_output, expected_sigmoid, 1e-9));

    // Test Tanh: f(x) = tanh(x)
    Tanh tanh_fn;
    double tanh_output = tanh_fn.activate(x);
    double expected_tanh = std::tanh(x);

    // Tanh output should be in range (-1, 1)
    RC_ASSERT(tanh_output > -1.0);
    RC_ASSERT(tanh_output < 1.0);
    RC_ASSERT(approxEqual(tanh_output, expected_tanh, 1e-9));

    // Test ReLU: f(x) = max(0, x)
    ReLU relu;
    double relu_output = relu.activate(x);
    double expected_relu = std::max(0.0, x);

    // ReLU output should be non-negative
    RC_ASSERT(relu_output >= 0.0);
    RC_ASSERT(approxEqual(relu_output, expected_relu, 1e-9));

    // Additional correctness checks
    if (x > 0) {
        RC_ASSERT(approxEqual(relu_output, x, 1e-9));
    } else {
        RC_ASSERT(approxEqual(relu_output, 0.0, 1e-9));
    }
}

// **Validates: Requirements 2.6**
// Feature: neural-network-framework, Property 6: Activation Function Derivative Verification
// Verify that analytical derivatives match numerical derivatives computed via finite differences within epsilon (1e-5)
RC_GTEST_PROP(ActivationPropertyTest, ActivationFunctionDerivativeVerification, ()) {
    // Generate random activation function and input value
    auto activation = *arbActivationFunction();
    auto x = *genDoubleInRange(-5.0, 5.0);

    // Compute analytical derivative
    double analytical = activation->derivative(x);

    // Compute numerical derivative via finite differences
    // f'(x) ≈ (f(x + h) - f(x - h)) / (2h)
    const double h = 1e-5;
    double f_plus = activation->activate(x + h);
    double f_minus = activation->activate(x - h);
    double numerical = (f_plus - f_minus) / (2.0 * h);

    // Verify they match within epsilon
    // Using 1e-4 tolerance to account for numerical approximation error
    RC_ASSERT(approxEqual(analytical, numerical, 1e-4));
}

// Additional property: Sigmoid derivative formula verification
// f'(x) = f(x) * (1 - f(x))
RC_GTEST_PROP(ActivationPropertyTest, SigmoidDerivativeFormula, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    Sigmoid sigmoid;
    double f_x = sigmoid.activate(x);
    double derivative = sigmoid.derivative(x);
    double expected_derivative = f_x * (1.0 - f_x);

    RC_ASSERT(approxEqual(derivative, expected_derivative, 1e-9));
}

// Additional property: Tanh derivative formula verification
// f'(x) = 1 - tanh²(x)
RC_GTEST_PROP(ActivationPropertyTest, TanhDerivativeFormula, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    Tanh tanh_fn;
    double f_x = tanh_fn.activate(x);
    double derivative = tanh_fn.derivative(x);
    double expected_derivative = 1.0 - f_x * f_x;

    RC_ASSERT(approxEqual(derivative, expected_derivative, 1e-9));
}

// Additional property: ReLU derivative correctness
// f'(x) = 1 if x > 0, else 0
RC_GTEST_PROP(ActivationPropertyTest, ReLUDerivativeCorrectness, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    ReLU relu;
    double derivative = relu.derivative(x);

    if (x > 0.0) {
        RC_ASSERT(approxEqual(derivative, 1.0, 1e-9));
    } else {
        RC_ASSERT(approxEqual(derivative, 0.0, 1e-9));
    }
}

// Additional property: Sigmoid is monotonically increasing
// For any x1 < x2, sigmoid(x1) < sigmoid(x2)
RC_GTEST_PROP(ActivationPropertyTest, SigmoidMonotonicity, ()) {
    double x1 = *genDoubleInRange(-10.0, 10.0);
    double x2 = *genDoubleInRange(-10.0, 10.0);

    // Ensure x1 < x2
    if (x1 > x2) {
        std::swap(x1, x2); // @suppress("Ambiguous problem")
    }

    // Skip if they're too close (within epsilon)
    RC_PRE(std::abs(x2 - x1) > 1e-6);

    Sigmoid sigmoid;
    double f_x1 = sigmoid.activate(x1);
    double f_x2 = sigmoid.activate(x2);

    RC_ASSERT(f_x1 < f_x2);
}

// Additional property: Tanh is monotonically increasing
// For any x1 < x2, tanh(x1) < tanh(x2)
RC_GTEST_PROP(ActivationPropertyTest, TanhMonotonicity, ()) {
    double x1 = *genDoubleInRange(-10.0, 10.0);
    double x2 = *genDoubleInRange(-10.0, 10.0);

    // Ensure x1 < x2
    if (x1 > x2) {
        std::swap(x1, x2); // @suppress("Ambiguous problem")
    }

    // Skip if they're too close (within epsilon)
    RC_PRE(std::abs(x2 - x1) > 1e-6);

    Tanh tanh_fn;
    double f_x1 = tanh_fn.activate(x1);
    double f_x2 = tanh_fn.activate(x2);

    RC_ASSERT(f_x1 < f_x2);
}

// Additional property: ReLU is monotonically non-decreasing
// For any x1 < x2, relu(x1) <= relu(x2)
RC_GTEST_PROP(ActivationPropertyTest, ReLUMonotonicity, ()) {
    double x1 = *genDoubleInRange(-10.0, 10.0);
    double x2 = *genDoubleInRange(-10.0, 10.0);

    // Ensure x1 < x2
    if (x1 > x2) {
        std::swap(x1, x2); // @suppress("Ambiguous problem")
    }

    // Skip if they're too close (within epsilon)
    RC_PRE(std::abs(x2 - x1) > 1e-6);

    ReLU relu;
    double f_x1 = relu.activate(x1);
    double f_x2 = relu.activate(x2);

    RC_ASSERT(f_x1 <= f_x2);
}

// Additional property: Sigmoid symmetry around 0.5
// sigmoid(-x) = 1 - sigmoid(x)
RC_GTEST_PROP(ActivationPropertyTest, SigmoidSymmetry, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    Sigmoid sigmoid;
    double f_x = sigmoid.activate(x);
    double f_neg_x = sigmoid.activate(-x);

    RC_ASSERT(approxEqual(f_neg_x, 1.0 - f_x, 1e-9));
}

// Additional property: Tanh is odd function
// tanh(-x) = -tanh(x)
RC_GTEST_PROP(ActivationPropertyTest, TanhOddFunction, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    Tanh tanh_fn;
    double f_x = tanh_fn.activate(x);
    double f_neg_x = tanh_fn.activate(-x);

    RC_ASSERT(approxEqual(f_neg_x, -f_x, 1e-9));
}

// Additional property: Activation function names are correct
RC_GTEST_PROP(ActivationPropertyTest, ActivationFunctionNames, ()) {
    Sigmoid sigmoid;
    Tanh tanh_fn;
    ReLU relu;

    RC_ASSERT(sigmoid.name() == "sigmoid");
    RC_ASSERT(tanh_fn.name() == "tanh");
    RC_ASSERT(relu.name() == "relu");
}

// Additional property: Sigmoid derivative is always non-negative
// Since f'(x) = f(x) * (1 - f(x)) and 0 < f(x) < 1, derivative is always positive
RC_GTEST_PROP(ActivationPropertyTest, SigmoidDerivativeNonNegative, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    Sigmoid sigmoid;
    double derivative = sigmoid.derivative(x);

    RC_ASSERT(derivative >= 0.0);
    RC_ASSERT(derivative <= 0.25);  // Maximum at x=0 is 0.25
}

// Additional property: Tanh derivative is always non-negative
// Since f'(x) = 1 - tanh²(x) and -1 < tanh(x) < 1, derivative is always positive
RC_GTEST_PROP(ActivationPropertyTest, TanhDerivativeNonNegative, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    Tanh tanh_fn;
    double derivative = tanh_fn.derivative(x);

    RC_ASSERT(derivative >= 0.0);
    RC_ASSERT(derivative <= 1.0);  // Maximum at x=0 is 1.0
}

// Additional property: ReLU derivative is binary (0 or 1)
RC_GTEST_PROP(ActivationPropertyTest, ReLUDerivativeBinary, ()) {
    auto x = *genDoubleInRange(-10.0, 10.0);

    ReLU relu;
    double derivative = relu.derivative(x);

    RC_ASSERT(approxEqual(derivative, 0.0, 1e-9) || approxEqual(derivative, 1.0, 1e-9));
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
