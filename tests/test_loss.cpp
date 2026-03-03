#include "loss.h"
#include <iostream>
#include <cmath>
#include <cassert>

// Helper function to check if two doubles are approximately equal
bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::abs(a - b) < epsilon;
}

// Helper function to check if two vectors are approximately equal
bool vectorApproxEqual(const Vector& a, const Vector& b, double epsilon = 1e-6) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!approxEqual(a[i], b[i], epsilon)) return false;
    }
    return true;
}

void testMSE() {
    std::cout << "Testing Mean Squared Error..." << std::endl;

    MeanSquaredError mse;

    // Test 1: Perfect prediction (zero loss)
    Vector predicted1 = {1.0, 2.0, 3.0};
    Vector target1 = {1.0, 2.0, 3.0};
    double loss1 = mse.compute(predicted1, target1);
    assert(approxEqual(loss1, 0.0));
    std::cout << "  ✓ Perfect prediction: loss = " << loss1 << std::endl;

    // Test 2: Known loss value
    // predicted = [1, 2, 3], target = [2, 3, 4]
    // diff = [-1, -1, -1], squared = [1, 1, 1], mean = 1.0
    Vector predicted2 = {1.0, 2.0, 3.0};
    Vector target2 = {2.0, 3.0, 4.0};
    double loss2 = mse.compute(predicted2, target2);
    assert(approxEqual(loss2, 1.0));
    std::cout << "  ✓ Known loss: loss = " << loss2 << std::endl;

    // Test 3: Gradient for perfect prediction (should be zero)
    Vector grad1 = mse.gradient(predicted1, target1);
    Vector expected_grad1 = {0.0, 0.0, 0.0};
    assert(vectorApproxEqual(grad1, expected_grad1));
    std::cout << "  ✓ Perfect prediction gradient: all zeros" << std::endl;

    // Test 4: Known gradient
    // gradient = (2/n) * (predicted - target) = (2/3) * [-1, -1, -1]
    Vector grad2 = mse.gradient(predicted2, target2);
    Vector expected_grad2 = {-2.0/3.0, -2.0/3.0, -2.0/3.0};
    assert(vectorApproxEqual(grad2, expected_grad2));
    std::cout << "  ✓ Known gradient: correct values" << std::endl;

    // Test 5: Dimension mismatch should throw
    Vector predicted3 = {1.0, 2.0};
    Vector target3 = {1.0, 2.0, 3.0};
    bool caught_exception = false;
    try {
        mse.compute(predicted3, target3);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::cout << "  ✓ Dimension mismatch detected: " << e.what() << std::endl;
    }
    assert(caught_exception);

    std::cout << "MSE tests passed!" << std::endl << std::endl;
}

void testCrossEntropy() {
    std::cout << "Testing Cross Entropy..." << std::endl;

    CrossEntropy ce;

    // Test 1: Perfect prediction (near-zero loss)
    Vector predicted1 = {1.0, 0.0, 0.0};
    Vector target1 = {1.0, 0.0, 0.0};
    double loss1 = ce.compute(predicted1, target1);
    // Loss should be very small (not exactly zero due to epsilon clamping)
    assert(loss1 < 0.01);
    std::cout << "  ✓ Perfect prediction: loss = " << loss1 << " (near zero)" << std::endl;

    // Test 2: Known loss value
    // predicted = [0.7, 0.2, 0.1], target = [1, 0, 0]
    // loss = -1*log(0.7) - 0*log(0.2) - 0*log(0.1) = -log(0.7) ≈ 0.357
    Vector predicted2 = {0.7, 0.2, 0.1};
    Vector target2 = {1.0, 0.0, 0.0};
    double loss2 = ce.compute(predicted2, target2);
    double expected_loss2 = -std::log(0.7);
    assert(approxEqual(loss2, expected_loss2, 1e-5));
    std::cout << "  ✓ Known loss: loss = " << loss2 << " (expected " << expected_loss2 << ")" << std::endl;

    // Test 3: Gradient test
    // gradient = -target / predicted
    // For target = [1, 0, 0], predicted = [0.7, 0.2, 0.1]
    // gradient = [-1/0.7, 0, 0] ≈ [-1.429, 0, 0]
    Vector grad2 = ce.gradient(predicted2, target2);
    Vector expected_grad2 = {-1.0/0.7, 0.0, 0.0};
    assert(vectorApproxEqual(grad2, expected_grad2, 1e-5));
    std::cout << "  ✓ Known gradient: correct values" << std::endl;

    // Test 4: Edge case - predicted value near zero (should use epsilon)
    Vector predicted3 = {0.0, 0.5, 0.5};
    Vector target3 = {1.0, 0.0, 0.0};
    double loss3 = ce.compute(predicted3, target3);
    // Should not be infinite due to epsilon handling
    assert(!std::isinf(loss3) && !std::isnan(loss3));
    std::cout << "  ✓ Edge case (predicted=0): loss = " << loss3 << " (finite)" << std::endl;

    // Test 5: Gradient edge case - predicted near zero
    Vector grad3 = ce.gradient(predicted3, target3);
    // Should not be infinite due to epsilon handling
    assert(!std::isinf(grad3[0]) && !std::isnan(grad3[0]));
    std::cout << "  ✓ Edge case gradient (predicted=0): finite values" << std::endl;

    // Test 6: Dimension mismatch should throw
    Vector predicted4 = {0.5, 0.5};
    Vector target4 = {1.0, 0.0, 0.0};
    bool caught_exception = false;
    try {
        ce.compute(predicted4, target4);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        std::cout << "  ✓ Dimension mismatch detected: " << e.what() << std::endl;
    }
    assert(caught_exception);

    std::cout << "Cross Entropy tests passed!" << std::endl << std::endl;
}

int main() {
    std::cout << "=== Loss Function Tests ===" << std::endl << std::endl;

    testMSE();
    testCrossEntropy();

    std::cout << "All loss function tests passed! ✓" << std::endl;
    return 0;
}
