#ifndef LOSS_H
#define LOSS_H

#include "vector.h"
#include <string>
#include <memory>

/**
 * Abstract base class for loss functions.
 * Loss functions measure the error between predicted and target outputs.
 */
class LossFunction {
public:
    virtual ~LossFunction() = default;

    /**
     * Compute the loss value between predicted and target vectors.
     * @param predicted The network's output
     * @param target The desired output
     * @return Scalar loss value
     */
    virtual double compute(const Vector& predicted, const Vector& target) const = 0;

    /**
     * Compute the gradient of the loss with respect to the predicted output.
     * @param predicted The network's output
     * @param target The desired output
     * @return Vector of gradients (same size as predicted/target)
     */
    virtual Vector gradient(const Vector& predicted, const Vector& target) const = 0;

    /**
     * Get the name of the loss function.
     * @return String identifier for the loss function
     */
    virtual std::string name() const = 0;
};

/**
 * Mean Squared Error loss function.
 * L = (1/n) * sum((predicted - target)^2)
 * Gradient: dL/dy = (2/n) * (predicted - target)
 */
class MeanSquaredError : public LossFunction {
public:
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
    std::string name() const override { return "mse"; }
};

/**
 * Cross-Entropy loss function.
 * L = -sum(target * log(predicted))
 * Gradient: dL/dy = -target / predicted
 *
 * Note: Includes epsilon for numerical stability to avoid log(0) and division by zero.
 */
class CrossEntropy : public LossFunction {
public:
    double compute(const Vector& predicted, const Vector& target) const override;
    Vector gradient(const Vector& predicted, const Vector& target) const override;
    std::string name() const override { return "cross_entropy"; }

private:
    static constexpr double EPSILON = 1e-10;  // Small value to prevent log(0)
};

#endif // LOSS_H
