#include "activation.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <limits>

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

void testSigmoidActivation() {
    std::cout << "Testing Sigmoid activation..." << std::endl;

    Sigmoid sigmoid;

    // Test name
    assert(sigmoid.name() == "sigmoid");

    // Test known values
    // sigmoid(0) = 0.5
    assert(approxEqual(sigmoid.activate(0.0), 0.5));

    // sigmoid(large positive) ≈ 1
    assert(sigmoid.activate(10.0) > 0.99);
    assert(sigmoid.activate(10.0) < 1.0);

    // sigmoid(large negative) ≈ 0
    assert(sigmoid.activate(-10.0) < 0.01);
    assert(sigmoid.activate(-10.0) > 0.0);

    // Test specific value: sigmoid(1) ≈ 0.7310585786
    double expected = 1.0 / (1.0 + std::exp(-1.0));
    assert(approxEqual(sigmoid.activate(1.0), expected));

    // Test range: output should always be in (0, 1)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        double output = sigmoid.activate(x);
        assert(output > 0.0 && output < 1.0);
    }

    // Test symmetry: sigmoid(-x) = 1 - sigmoid(x)
    for (double x = -5.0; x <= 5.0; x += 1.0) {
        double f_x = sigmoid.activate(x);
        double f_neg_x = sigmoid.activate(-x);
        assert(approxEqual(f_neg_x, 1.0 - f_x));
    }

    std::cout << "  ✓ Sigmoid activation passed" << std::endl;
}

void testSigmoidDerivative() {
    std::cout << "Testing Sigmoid derivative..." << std::endl;

    Sigmoid sigmoid;

    // Test derivative formula: f'(x) = f(x) * (1 - f(x))
    for (double x = -5.0; x <= 5.0; x += 0.5) {
        double f_x = sigmoid.activate(x);
        double derivative = sigmoid.derivative(x);
        double expected = f_x * (1.0 - f_x);
        assert(approxEqual(derivative, expected));
    }

    // Test derivative at x=0: f'(0) = 0.25
    assert(approxEqual(sigmoid.derivative(0.0), 0.25));

    // Test derivative is always non-negative
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        assert(sigmoid.derivative(x) >= 0.0);
    }

    // Test derivative maximum is 0.25 (at x=0)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        assert(sigmoid.derivative(x) <= 0.25 + 1e-9);
    }

    std::cout << "  ✓ Sigmoid derivative passed" << std::endl;
}

void testTanhActivation() {
    std::cout << "Testing Tanh activation..." << std::endl;

    Tanh tanh_fn;

    // Test name
    assert(tanh_fn.name() == "tanh");

    // Test known values
    // tanh(0) = 0
    assert(approxEqual(tanh_fn.activate(0.0), 0.0));

    // tanh(large positive) ≈ 1
    assert(tanh_fn.activate(10.0) > 0.99);
    assert(tanh_fn.activate(10.0) < 1.0);

    // tanh(large negative) ≈ -1
    assert(tanh_fn.activate(-10.0) < -0.99);
    assert(tanh_fn.activate(-10.0) > -1.0);

    // Test specific value: tanh(1) ≈ 0.7615941559
    double expected = std::tanh(1.0);
    assert(approxEqual(tanh_fn.activate(1.0), expected));

    // Test range: output should always be in (-1, 1)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        double output = tanh_fn.activate(x);
        assert(output > -1.0 && output < 1.0);
    }

    // Test odd function property: tanh(-x) = -tanh(x)
    for (double x = -5.0; x <= 5.0; x += 1.0) {
        double f_x = tanh_fn.activate(x);
        double f_neg_x = tanh_fn.activate(-x);
        assert(approxEqual(f_neg_x, -f_x));
    }

    std::cout << "  ✓ Tanh activation passed" << std::endl;
}

void testTanhDerivative() {
    std::cout << "Testing Tanh derivative..." << std::endl;

    Tanh tanh_fn;

    // Test derivative formula: f'(x) = 1 - tanh²(x)
    for (double x = -5.0; x <= 5.0; x += 0.5) {
        double f_x = tanh_fn.activate(x);
        double derivative = tanh_fn.derivative(x);
        double expected = 1.0 - f_x * f_x;
        assert(approxEqual(derivative, expected));
    }

    // Test derivative at x=0: f'(0) = 1
    assert(approxEqual(tanh_fn.derivative(0.0), 1.0));

    // Test derivative is always non-negative
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        assert(tanh_fn.derivative(x) >= 0.0);
    }

    // Test derivative maximum is 1.0 (at x=0)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        assert(tanh_fn.derivative(x) <= 1.0 + 1e-9);
    }

    std::cout << "  ✓ Tanh derivative passed" << std::endl;
}

void testReLUActivation() {
    std::cout << "Testing ReLU activation..." << std::endl;

    ReLU relu;

    // Test name
    assert(relu.name() == "relu");

    // Test known values
    // relu(0) = 0
    assert(approxEqual(relu.activate(0.0), 0.0));

    // relu(positive) = positive
    assert(approxEqual(relu.activate(5.0), 5.0));
    assert(approxEqual(relu.activate(0.5), 0.5));
    assert(approxEqual(relu.activate(100.0), 100.0));

    // relu(negative) = 0
    assert(approxEqual(relu.activate(-5.0), 0.0));
    assert(approxEqual(relu.activate(-0.5), 0.0));
    assert(approxEqual(relu.activate(-100.0), 0.0));

    // Test range: output should always be non-negative
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        double output = relu.activate(x);
        assert(output >= 0.0);
    }

    // Test identity for positive values
    for (double x = 0.1; x <= 10.0; x += 0.5) {
        assert(approxEqual(relu.activate(x), x));
    }

    // Test zero for negative values
    for (double x = -10.0; x < 0.0; x += 0.5) {
        assert(approxEqual(relu.activate(x), 0.0));
    }

    std::cout << "  ✓ ReLU activation passed" << std::endl;
}

void testReLUDerivative() {
    std::cout << "Testing ReLU derivative..." << std::endl;

    ReLU relu;

    // Test derivative for positive values: f'(x) = 1 for x > 0
    for (double x = 0.1; x <= 10.0; x += 0.5) {
        assert(approxEqual(relu.derivative(x), 1.0));
    }

    // Test derivative for negative values: f'(x) = 0 for x < 0
    for (double x = -10.0; x < 0.0; x += 0.5) {
        assert(approxEqual(relu.derivative(x), 0.0));
    }

    // Test derivative at x=0: f'(0) = 0 (by convention)
    assert(approxEqual(relu.derivative(0.0), 0.0));

    // Test derivative is binary (0 or 1)
    for (double x = -10.0; x <= 10.0; x += 0.5) {
        double derivative = relu.derivative(x);
        assert(approxEqual(derivative, 0.0) || approxEqual(derivative, 1.0));
    }

    std::cout << "  ✓ ReLU derivative passed" << std::endl;
}

void testMonotonicity() {
    std::cout << "Testing monotonicity properties..." << std::endl;

    Sigmoid sigmoid;
    Tanh tanh_fn;
    ReLU relu;

    // Test that all activation functions are monotonically increasing
    for (double x1 = -5.0; x1 < 5.0; x1 += 1.0) {
        double x2 = x1 + 0.5;

        // Sigmoid is strictly increasing
        assert(sigmoid.activate(x1) < sigmoid.activate(x2));

        // Tanh is strictly increasing
        assert(tanh_fn.activate(x1) < tanh_fn.activate(x2));

        // ReLU is non-decreasing
        assert(relu.activate(x1) <= relu.activate(x2));
    }

    std::cout << "  ✓ Monotonicity properties passed" << std::endl;
}

void testNumericalStability() {
    std::cout << "Testing numerical stability..." << std::endl;

    Sigmoid sigmoid;
    Tanh tanh_fn;
    ReLU relu;

    // Test with very large positive values
    double large_pos = 100.0;
    assert(!std::isnan(sigmoid.activate(large_pos)));
    assert(!std::isinf(sigmoid.activate(large_pos)));
    assert(!std::isnan(tanh_fn.activate(large_pos)));
    assert(!std::isinf(tanh_fn.activate(large_pos)));
    assert(!std::isnan(relu.activate(large_pos)));
    assert(!std::isinf(relu.activate(large_pos)));

    // Test with very large negative values
    double large_neg = -100.0;
    assert(!std::isnan(sigmoid.activate(large_neg)));
    assert(!std::isinf(sigmoid.activate(large_neg)));
    assert(!std::isnan(tanh_fn.activate(large_neg)));
    assert(!std::isinf(tanh_fn.activate(large_neg)));
    assert(!std::isnan(relu.activate(large_neg)));
    assert(!std::isinf(relu.activate(large_neg)));

    // Test derivatives with extreme values
    assert(!std::isnan(sigmoid.derivative(large_pos)));
    assert(!std::isnan(sigmoid.derivative(large_neg)));
    assert(!std::isnan(tanh_fn.derivative(large_pos)));
    assert(!std::isnan(tanh_fn.derivative(large_neg)));
    assert(!std::isnan(relu.derivative(large_pos)));
    assert(!std::isnan(relu.derivative(large_neg)));

    std::cout << "  ✓ Numerical stability passed" << std::endl;
}

void testDerivativeVsNumerical() {
    std::cout << "Testing analytical vs numerical derivatives..." << std::endl;

    Sigmoid sigmoid;
    Tanh tanh_fn;
    ReLU relu;

    const double h = 1e-5;
    const double tolerance = 1e-4;

    // Test Sigmoid
    for (double x = -5.0; x <= 5.0; x += 1.0) {
        double analytical = sigmoid.derivative(x);
        double numerical = (sigmoid.activate(x + h) - sigmoid.activate(x - h)) / (2.0 * h);
        assert(approxEqual(analytical, numerical, tolerance));
    }

    // Test Tanh
    for (double x = -5.0; x <= 5.0; x += 1.0) {
        double analytical = tanh_fn.derivative(x);
        double numerical = (tanh_fn.activate(x + h) - tanh_fn.activate(x - h)) / (2.0 * h);
        assert(approxEqual(analytical, numerical, tolerance));
    }

    // Test ReLU (skip x=0 where derivative is discontinuous)
    for (double x = -5.0; x <= 5.0; x += 1.0) {
        if (std::abs(x) < 0.1) continue; // Skip near zero
        double analytical = relu.derivative(x);
        double numerical = (relu.activate(x + h) - relu.activate(x - h)) / (2.0 * h);
        assert(approxEqual(analytical, numerical, tolerance));
    }

    std::cout << "  ✓ Analytical vs numerical derivatives passed" << std::endl;
}

int main() {
    std::cout << "Running Activation Function tests..." << std::endl;
    std::cout << "===================================" << std::endl;

    testSigmoidActivation();
    testSigmoidDerivative();
    testTanhActivation();
    testTanhDerivative();
    testReLUActivation();
    testReLUDerivative();
    testMonotonicity();
    testNumericalStability();
    testDerivativeVsNumerical();

    std::cout << std::endl;
    std::cout << "All Activation Function tests passed! ✓" << std::endl;

    return 0;
}
