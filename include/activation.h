#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <string>
#include <cmath>

/**
 * Abstract base class for activation functions.
 * Defines the interface for activation functions used in neural network layers.
 * Each activation function must provide:
 * - activate(x): compute the activation value
 * - derivative(x): compute the derivative for backpropagation
 * - name(): return the function name for serialization
 */
class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    /**
     * Compute the activation function output for a given input.
     * @param x Input value
     * @return Activation function output
     */
    virtual double activate(double x) const = 0;

    /**
     * Compute the derivative of the activation function.
     * @param x Input value
     * @return Derivative value at x
     */
    virtual double derivative(double x) const = 0;

    /**
     * Get the name of the activation function.
     * @return Name string (e.g., "sigmoid", "tanh", "relu")
     */
    virtual std::string name() const = 0;
};

/**
 * Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 * Derivative: f'(x) = f(x) * (1 - f(x))
 * Range: (0, 1)
 * Used for binary classification and when outputs need to be probabilities.
 */
class Sigmoid : public ActivationFunction {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string name() const override { return "sigmoid"; }
};

/**
 * Hyperbolic tangent activation function: f(x) = tanh(x)
 * Derivative: f'(x) = 1 - tanh²(x)
 * Range: (-1, 1)
 * Zero-centered, often works better than sigmoid in hidden layers.
 */
class Tanh : public ActivationFunction {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string name() const override { return "tanh"; }
};

/**
 * Rectified Linear Unit activation function: f(x) = max(0, x)
 * Derivative: f'(x) = 1 if x > 0, else 0
 * Range: [0, ∞)
 * Most popular for deep networks, helps avoid vanishing gradient problem.
 */
class ReLU : public ActivationFunction {
public:
    double activate(double x) const override;
    double derivative(double x) const override;
    std::string name() const override { return "relu"; }
};

#endif // ACTIVATION_H
