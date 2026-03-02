#include "activation.h"
#include <cmath>
#include <algorithm>

// ============================================================================
// Sigmoid Implementation
// ============================================================================

/**
 * Sigmoid activation: f(x) = 1 / (1 + e^(-x))
 *
 * Properties:
 * - Smooth, differentiable everywhere
 * - Output range: (0, 1)
 * - Saturates for large |x|, which can cause vanishing gradients
 *
 * @param x Input value
 * @return Sigmoid output in range (0, 1)
 */
double Sigmoid::activate(double x) const {
    // Use numerically stable formulation to avoid overflow
    // For large positive x: 1/(1+e^-x) ≈ 1
    // For large negative x: e^x/(1+e^x) avoids overflow
    if (x >= 0) {
        return 1.0 / (1.0 + std::exp(-x));
    } else {
        double exp_x = std::exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

/**
 * Sigmoid derivative: f'(x) = f(x) * (1 - f(x))
 *
 * This is computed using the sigmoid value itself, which is efficient
 * during backpropagation since we already have the activation output.
 *
 * @param x Input value
 * @return Derivative value
 */
double Sigmoid::derivative(double x) const {
    double sig = activate(x);
    return sig * (1.0 - sig);
}

// ============================================================================
// Tanh Implementation
// ============================================================================

/**
 * Hyperbolic tangent activation: f(x) = tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 *
 * Properties:
 * - Smooth, differentiable everywhere
 * - Output range: (-1, 1)
 * - Zero-centered (unlike sigmoid), which can help with training
 * - Still suffers from vanishing gradients for large |x|
 *
 * @param x Input value
 * @return Tanh output in range (-1, 1)
 */
double Tanh::activate(double x) const {
    return std::tanh(x);
}

/**
 * Tanh derivative: f'(x) = 1 - tanh²(x)
 *
 * Maximum derivative is 1 at x=0, decreases as |x| increases.
 *
 * @param x Input value
 * @return Derivative value in range (0, 1]
 */
double Tanh::derivative(double x) const {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

// ============================================================================
// ReLU Implementation
// ============================================================================

/**
 * Rectified Linear Unit activation: f(x) = max(0, x)
 *
 * Properties:
 * - Simple, computationally efficient
 * - Output range: [0, ∞)
 * - Non-saturating for positive values (helps avoid vanishing gradients)
 * - Not differentiable at x=0 (we use convention: derivative = 0 at x=0)
 * - Can suffer from "dying ReLU" problem (neurons stuck at 0)
 *
 * @param x Input value
 * @return ReLU output: x if x > 0, else 0
 */
double ReLU::activate(double x) const {
    return std::max(0.0, x);
}

/**
 * ReLU derivative: f'(x) = 1 if x > 0, else 0
 *
 * Note: At x=0, the derivative is technically undefined (left derivative is 0,
 * right derivative is 1). By convention, we use 0 at x=0, which works well
 * in practice for gradient descent.
 *
 * @param x Input value
 * @return 1.0 if x > 0, else 0.0
 */
double ReLU::derivative(double x) const {
    return x > 0.0 ? 1.0 : 0.0;
}
