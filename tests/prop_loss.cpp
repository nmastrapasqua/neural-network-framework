#include "loss.h"
#include "vector.h"
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

// Generator for vectors of specific size with values in range
rc::Gen<Vector> arbVectorOfSize(size_t size, double min = -10.0, double max = 10.0) {
    return rc::gen::apply([size, min, max](const std::vector<double>& values) {
        Vector v(size);
        for (size_t i = 0; i < size && i < values.size(); ++i) {
            v[i] = values[i];
        }
        return v;
    }, rc::gen::container<std::vector<double>>(size, genDoubleInRange(min, max)));
}

// Generator for probability vectors (values in [0, 1] that sum to ~1)
rc::Gen<Vector> arbProbabilityVector(size_t size) {
    return rc::gen::apply([size](const std::vector<double>& values) {
        Vector v(size);
        double sum = 0.0;
        for (size_t i = 0; i < size && i < values.size(); ++i) {
            v[i] = std::abs(values[i]);
            sum += v[i];
        }
        // Normalize to sum to 1
        if (sum > 1e-10) {
            for (size_t i = 0; i < size; ++i) {
                v[i] /= sum;
            }
        }
        return v;
    }, rc::gen::container<std::vector<double>>(size, genDoubleInRange(0.0, 1.0)));
}

// Generator for loss functions
rc::Gen<std::shared_ptr<LossFunction>> arbLossFunction() {
    return rc::gen::map(
        rc::gen::inRange(0, 2),
        [](int choice) -> std::shared_ptr<LossFunction> {
            switch (choice) {
                case 0: return std::make_shared<MeanSquaredError>();
                case 1: return std::make_shared<CrossEntropy>();
                default: return std::make_shared<MeanSquaredError>();
            }
        }
    );
}

// **Validates: Requirements 5.1, 5.2, 5.4**
// Feature: neural-network-framework, Property 16: Loss Function Correctness
// Verify that MSE and Cross-Entropy produce mathematically correct outputs according to their definitions
RC_GTEST_PROP(LossPropertyTest, LossFunctionCorrectness, ()) {
    // Generate random size for vectors
    auto size = *rc::gen::inRange<size_t>(1, 20);

    // Generate predicted and target vectors
    auto predicted = *arbVectorOfSize(size, -5.0, 5.0);
    auto target = *arbVectorOfSize(size, -5.0, 5.0);

    // Test Mean Squared Error: L = (1/n) * sum((predicted - target)^2)
    MeanSquaredError mse;
    double mse_loss = mse.compute(predicted, target);

    // Compute expected MSE manually
    double expected_mse = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double diff = predicted[i] - target[i];
        expected_mse += diff * diff;
    }
    expected_mse /= static_cast<double>(size);

    // MSE should be non-negative
    RC_ASSERT(mse_loss >= 0.0);
    RC_ASSERT(approxEqual(mse_loss, expected_mse, 1e-9));

    // Test Cross-Entropy with probability vectors
    // Generate probability vectors for cross-entropy (values in [0, 1])
    auto pred_prob = *arbProbabilityVector(size);
    auto target_prob = *arbProbabilityVector(size);

    CrossEntropy ce;
    double ce_loss = ce.compute(pred_prob, target_prob);

    // Compute expected cross-entropy manually: -sum(target * log(predicted))
    double expected_ce = 0.0;
    const double EPSILON = 1e-10;
    for (size_t i = 0; i < size; ++i) {
        double pred_clamped = std::max(EPSILON, std::min(1.0, pred_prob[i]));
        expected_ce -= target_prob[i] * std::log(pred_clamped);
    }

    // Cross-entropy should be non-negative for probability distributions
    RC_ASSERT(ce_loss >= -1e-9);  // Allow small negative due to floating point
    RC_ASSERT(approxEqual(ce_loss, expected_ce, 1e-6));
}

// **Validates: Requirements 5.5**
// Feature: neural-network-framework, Property 17: Loss Gradient Verification
// Verify that analytical gradients match numerical gradients computed via finite differences within epsilon
RC_GTEST_PROP(LossPropertyTest, LossGradientVerification, ()) {
    // Generate random size for vectors
    auto size = *rc::gen::inRange<size_t>(1, 10);

    // Generate predicted and target vectors
    auto predicted = *arbVectorOfSize(size, -3.0, 3.0);
    auto target = *arbVectorOfSize(size, -3.0, 3.0);

    // Test MSE gradient
    MeanSquaredError mse;
    Vector mse_grad = mse.gradient(predicted, target);

    // Verify gradient dimensions
    RC_ASSERT(mse_grad.size() == size);

    // Compute numerical gradient via finite differences
    const double h = 1e-5;
    for (size_t i = 0; i < size; ++i) {
        // Perturb predicted[i] by +h
        Vector pred_plus = predicted;
        pred_plus[i] += h;
        double loss_plus = mse.compute(pred_plus, target);

        // Perturb predicted[i] by -h
        Vector pred_minus = predicted;
        pred_minus[i] -= h;
        double loss_minus = mse.compute(pred_minus, target);

        // Numerical gradient: (f(x+h) - f(x-h)) / (2h)
        double numerical_grad = (loss_plus - loss_minus) / (2.0 * h);

        // Compare with analytical gradient
        RC_ASSERT(approxEqual(mse_grad[i], numerical_grad, 1e-4));
    }

    // Test Cross-Entropy gradient with probability vectors
    auto pred_prob = *arbProbabilityVector(size);
    auto target_prob = *arbProbabilityVector(size);

    // Ensure predicted values are in a stable range [0.05, 0.95] for numerical gradient computation
    // This avoids issues with log near 0 and 1
    for (size_t i = 0; i < size; ++i) {
        pred_prob[i] = std::max(0.05, std::min(0.95, pred_prob[i]));
    }

    // Skip if target has very small values that would make gradient verification unstable
    bool has_significant_target = false;
    for (size_t i = 0; i < size; ++i) {
        if (target_prob[i] > 0.01) {
            has_significant_target = true;
            break;
        }
    }
    RC_PRE(has_significant_target);

    CrossEntropy ce;
    Vector ce_grad = ce.gradient(pred_prob, target_prob);

    // Verify gradient dimensions
    RC_ASSERT(ce_grad.size() == size);

    // Compute numerical gradient for cross-entropy
    // Use adaptive step size based on predicted value
    for (size_t i = 0; i < size; ++i) {
        // Use smaller step size for more accurate numerical gradient
        double step = h * pred_prob[i];

        Vector pred_plus = pred_prob;
        pred_plus[i] += step;
        // Clamp to valid range
        pred_plus[i] = std::min(0.99, pred_plus[i]);
        double loss_plus = ce.compute(pred_plus, target_prob);

        Vector pred_minus = pred_prob;
        pred_minus[i] -= step;
        // Clamp to valid range
        pred_minus[i] = std::max(0.01, pred_minus[i]);
        double loss_minus = ce.compute(pred_minus, target_prob);

        double numerical_grad = (loss_plus - loss_minus) / (2.0 * step);

        // Use adaptive tolerance based on gradient magnitude
        double tolerance = std::max(1e-2, std::abs(ce_grad[i]) * 0.05);

        // Only check gradient for components where target is significant
        if (target_prob[i] > 0.01) {
            RC_ASSERT(approxEqual(ce_grad[i], numerical_grad, tolerance));
        }
    }
}

// Additional property: MSE gradient formula verification
// Gradient: (2/n) * (predicted - target)
RC_GTEST_PROP(LossPropertyTest, MSEGradientFormula, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto predicted = *arbVectorOfSize(size, -5.0, 5.0);
    auto target = *arbVectorOfSize(size, -5.0, 5.0);

    MeanSquaredError mse;
    Vector grad = mse.gradient(predicted, target);

    // Verify gradient formula
    double scale = 2.0 / static_cast<double>(size);
    for (size_t i = 0; i < size; ++i) {
        double expected_grad = scale * (predicted[i] - target[i]);
        RC_ASSERT(approxEqual(grad[i], expected_grad, 1e-9));
    }
}

// Additional property: MSE is zero when predicted equals target
RC_GTEST_PROP(LossPropertyTest, MSEZeroForPerfectPrediction, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto target = *arbVectorOfSize(size, -5.0, 5.0);

    MeanSquaredError mse;
    double loss = mse.compute(target, target);

    RC_ASSERT(approxEqual(loss, 0.0, 1e-9));
}

// Additional property: MSE gradient is zero when predicted equals target
RC_GTEST_PROP(LossPropertyTest, MSEGradientZeroForPerfectPrediction, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto target = *arbVectorOfSize(size, -5.0, 5.0);

    MeanSquaredError mse;
    Vector grad = mse.gradient(target, target);

    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(grad[i], 0.0, 1e-9));
    }
}

// Additional property: MSE is symmetric
// MSE(a, b) = MSE(b, a)
RC_GTEST_PROP(LossPropertyTest, MSESymmetry, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto a = *arbVectorOfSize(size, -5.0, 5.0);
    auto b = *arbVectorOfSize(size, -5.0, 5.0);

    MeanSquaredError mse;
    double loss_ab = mse.compute(a, b);
    double loss_ba = mse.compute(b, a);

    RC_ASSERT(approxEqual(loss_ab, loss_ba, 1e-9));
}

// Additional property: MSE is always non-negative
RC_GTEST_PROP(LossPropertyTest, MSENonNegative, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto predicted = *arbVectorOfSize(size, -10.0, 10.0);
    auto target = *arbVectorOfSize(size, -10.0, 10.0);

    MeanSquaredError mse;
    double loss = mse.compute(predicted, target);

    RC_ASSERT(loss >= 0.0);
}

// Additional property: Cross-Entropy gradient formula verification
// Gradient: -target / predicted
RC_GTEST_PROP(LossPropertyTest, CrossEntropyGradientFormula, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto pred_prob = *arbProbabilityVector(size);
    auto target_prob = *arbProbabilityVector(size);

    // Ensure predicted values are not too close to 0
    for (size_t i = 0; i < size; ++i) {
        if (pred_prob[i] < 0.01) {
            pred_prob[i] = 0.01;
        }
    }

    CrossEntropy ce;
    Vector grad = ce.gradient(pred_prob, target_prob);

    // Verify gradient formula
    const double EPSILON = 1e-10;
    for (size_t i = 0; i < size; ++i) {
        double pred_clamped = std::max(EPSILON, pred_prob[i]);
        double expected_grad = -target_prob[i] / pred_clamped;
        RC_ASSERT(approxEqual(grad[i], expected_grad, 1e-6));
    }
}

// Additional property: Cross-Entropy is minimized when predicted equals target
RC_GTEST_PROP(LossPropertyTest, CrossEntropyMinimizedForPerfectPrediction, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 10);
    auto target_prob = *arbProbabilityVector(size);

    CrossEntropy ce;

    // Loss when predicted = target
    double loss_perfect = ce.compute(target_prob, target_prob);

    // Loss when predicted is different from target
    auto other_prob = *arbProbabilityVector(size);

    // Skip if they're too similar
    double diff = 0.0;
    for (size_t i = 0; i < size; ++i) {
        diff += std::abs(target_prob[i] - other_prob[i]);
    }
    RC_PRE(diff > 0.1);

    double loss_other = ce.compute(other_prob, target_prob);

    // Perfect prediction should have lower or equal loss
    RC_ASSERT(loss_perfect <= loss_other + 1e-6);
}

// Additional property: Loss function names are correct
RC_GTEST_PROP(LossPropertyTest, LossFunctionNames, ()) {
    MeanSquaredError mse;
    CrossEntropy ce;

    RC_ASSERT(mse.name() == "mse");
    RC_ASSERT(ce.name() == "cross_entropy");
}

// Additional property: MSE scales quadratically with error magnitude
// If we double the error, MSE should quadruple
RC_GTEST_PROP(LossPropertyTest, MSEQuadraticScaling, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 10);
    auto target = *arbVectorOfSize(size, -5.0, 5.0);

    // Create predicted with some error
    Vector predicted(size);
    for (size_t i = 0; i < size; ++i) {
        predicted[i] = target[i] + 1.0;  // Error of 1.0
    }

    // Create predicted with double the error
    Vector predicted_double(size);
    for (size_t i = 0; i < size; ++i) {
        predicted_double[i] = target[i] + 2.0;  // Error of 2.0
    }

    MeanSquaredError mse;
    double loss1 = mse.compute(predicted, target);
    double loss2 = mse.compute(predicted_double, target);

    // loss2 should be approximately 4 * loss1
    RC_ASSERT(approxEqual(loss2, 4.0 * loss1, 1e-6));
}

// Additional property: Dimension mismatch detection for compute
RC_GTEST_PROP(LossPropertyTest, DimensionMismatchDetectionCompute, ()) {
    auto size1 = *rc::gen::inRange<size_t>(1, 10);
    auto size2 = *rc::gen::inRange<size_t>(11, 20);

    auto predicted = *arbVectorOfSize(size1);
    auto target = *arbVectorOfSize(size2);

    MeanSquaredError mse;

    // Should throw exception for dimension mismatch
    bool caught_exception = false;
    try {
        mse.compute(predicted, target);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        // Verify error message contains useful information
        std::string msg = e.what();
        RC_ASSERT(msg.find("size") != std::string::npos);
    }

    RC_ASSERT(caught_exception);
}

// Additional property: Dimension mismatch detection for gradient
RC_GTEST_PROP(LossPropertyTest, DimensionMismatchDetectionGradient, ()) {
    auto size1 = *rc::gen::inRange<size_t>(1, 10);
    auto size2 = *rc::gen::inRange<size_t>(11, 20);

    auto predicted = *arbVectorOfSize(size1);
    auto target = *arbVectorOfSize(size2);

    MeanSquaredError mse;

    // Should throw exception for dimension mismatch
    bool caught_exception = false;
    try {
        mse.gradient(predicted, target);
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        // Verify error message contains useful information
        std::string msg = e.what();
        RC_ASSERT(msg.find("size") != std::string::npos);
    }

    RC_ASSERT(caught_exception);
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
