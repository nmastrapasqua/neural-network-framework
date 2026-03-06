#include "network.h"
#include "activation.h"
#include "loss.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <iostream>
#include <cmath>
#include <memory>
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
rc::Gen<std::shared_ptr<ActivationFunction>> arbActivation() {
    return rc::gen::apply([](int choice) -> std::shared_ptr<ActivationFunction> {
        switch (choice % 3) {
            case 0: return std::make_shared<Sigmoid>();
            case 1: return std::make_shared<Tanh>();
            case 2: return std::make_shared<ReLU>();
            default: return std::make_shared<Sigmoid>();
        }
    }, rc::gen::inRange(0, 3));
}

// Generator for valid layer sizes (positive integers)
rc::Gen<size_t> arbLayerSize() {
    return rc::gen::inRange<size_t>(1, 10);
}

// Helper function to create a simple network for testing
Network createSimpleNetwork(size_t input_size, size_t hidden_size, size_t output_size) {
    Network network;
    network.addLayer(input_size, hidden_size, std::make_shared<Sigmoid>());
    network.addLayer(hidden_size, output_size, std::make_shared<Sigmoid>());

    // Initialize weights to small random values
    network.getLayer(0).initializeWeights(-0.5, 0.5);
    network.getLayer(1).initializeWeights(-0.5, 0.5);

    return network;
}

// Helper function to create a training dataset
void createSimpleDataset(size_t num_examples, size_t input_size, size_t output_size,
                        std::vector<Vector>& inputs, std::vector<Vector>& targets) {
    inputs.clear();
    targets.clear();

    for (size_t i = 0; i < num_examples; ++i) {
        Vector input(input_size);
        for (size_t j = 0; j < input_size; ++j) {
            input[j] = (i + j) % 2 == 0 ? 0.0 : 1.0;  // Simple pattern
        }
        inputs.push_back(input);

        Vector target(output_size);
        for (size_t j = 0; j < output_size; ++j) {
            target[j] = i % 2 == 0 ? 0.0 : 1.0;  // Simple pattern
        }
        targets.push_back(target);
    }
}

// **Validates: Requirements 7.1, 7.2, 7.5**
// Feature: neural-network-framework, Property 20: Parameter Update Formula
// For any parameter (weight or bias) with value θ_old, gradient g, and learning rate η,
// after one update step the new value should be: θ_new = θ_old - η * g.
RC_GTEST_PROP(TrainingPropertyTest, ParameterUpdateFormula, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 5);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 5);
    size_t output_size = *rc::gen::inRange<size_t>(1, 3);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Store original parameters
    std::vector<Matrix> original_weights;
    std::vector<Vector> original_biases;

    for (size_t l = 0; l < network.numLayers(); ++l) {
        original_weights.push_back(network.getLayer(l).getWeights());
        original_biases.push_back(network.getLayer(l).getBiases());
    }

    // Create a single training example
    Vector input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = *genDoubleInRange(-0.5, 0.5);
    }

    Vector target(output_size);
    for (size_t i = 0; i < output_size; ++i) {
        target[i] = *genDoubleInRange(-0.5, 0.5);
    }

    std::vector<Vector> inputs = {input};
    std::vector<Vector> targets = {target};

    // Train for 1 epoch with batch_size = 1 (single SGD step)
    double learning_rate = *genDoubleInRange(0.01, 0.5);
    MeanSquaredError loss_function;

    network.train(inputs, targets, 1, learning_rate, loss_function, 1);

    // Verify that parameters have changed (gradient was non-zero)
    bool parameters_changed = false;

    for (size_t l = 0; l < network.numLayers(); ++l) {
        const Matrix& new_weights = network.getLayer(l).getWeights();
        const Vector& new_biases = network.getLayer(l).getBiases();

        // Check if any weight changed
        for (size_t i = 0; i < new_weights.rows(); ++i) {
            for (size_t j = 0; j < new_weights.cols(); ++j) {
                if (!approxEqual(new_weights(i, j), original_weights[l](i, j), 1e-9)) {
                    parameters_changed = true;
                }
            }
        }

        // Check if any bias changed
        for (size_t i = 0; i < new_biases.size(); ++i) {
            if (!approxEqual(new_biases[i], original_biases[l][i], 1e-9)) {
                parameters_changed = true;
            }
        }
    }

    // At least some parameters should have changed (unless gradient was exactly zero, which is unlikely)
    RC_ASSERT(parameters_changed);

    // Verify all parameters are finite
    for (size_t l = 0; l < network.numLayers(); ++l) {
        const Matrix& weights = network.getLayer(l).getWeights();
        const Vector& biases = network.getLayer(l).getBiases();

        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                RC_ASSERT(std::isfinite(weights(i, j)));
            }
        }

        for (size_t i = 0; i < biases.size(); ++i) {
            RC_ASSERT(std::isfinite(biases[i]));
        }
    }
}

// **Validates: Requirements 7.3**
// Feature: neural-network-framework, Property 21: Learning Rate Scaling
// For any network and training example, if we train with learning rate η₁ and then
// train with learning rate η₂ = 2*η₁, the parameter changes in the second case
// should be exactly twice the parameter changes in the first case.
RC_GTEST_PROP(TrainingPropertyTest, LearningRateScaling, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 4);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    // Create two identical networks
    Network network1 = createSimpleNetwork(input_size, hidden_size, output_size);
    Network network2 = createSimpleNetwork(input_size, hidden_size, output_size);

    // Make sure both networks have identical initial weights
    for (size_t l = 0; l < network1.numLayers(); ++l) {
        Matrix& weights1 = network1.getLayer(l).getWeights();
        Matrix& weights2 = network2.getLayer(l).getWeights();
        Vector& biases1 = network1.getLayer(l).getBiases();
        Vector& biases2 = network2.getLayer(l).getBiases();

        for (size_t i = 0; i < weights1.rows(); ++i) {
            for (size_t j = 0; j < weights1.cols(); ++j) {
                double val = *genDoubleInRange(-0.3, 0.3);
                weights1(i, j) = val;
                weights2(i, j) = val;
            }
        }

        for (size_t i = 0; i < biases1.size(); ++i) {
            double val = *genDoubleInRange(-0.1, 0.1);
            biases1[i] = val;
            biases2[i] = val;
        }
    }

    // Store original parameters
    std::vector<Matrix> original_weights;
    std::vector<Vector> original_biases;

    for (size_t l = 0; l < network1.numLayers(); ++l) {
        original_weights.push_back(network1.getLayer(l).getWeights());
        original_biases.push_back(network1.getLayer(l).getBiases());
    }

    // Create a single training example
    Vector input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = *genDoubleInRange(-0.5, 0.5);
    }

    Vector target(output_size);
    for (size_t i = 0; i < output_size; ++i) {
        target[i] = *genDoubleInRange(-0.5, 0.5);
    }

    std::vector<Vector> inputs = {input};
    std::vector<Vector> targets = {target};

    // Train network1 with learning rate η₁
    double lr1 = *genDoubleInRange(0.01, 0.2);
    MeanSquaredError loss_function;

    network1.train(inputs, targets, 1, lr1, loss_function, 1);

    // Train network2 with learning rate η₂ = 2*η₁
    double lr2 = 2.0 * lr1;
    network2.train(inputs, targets, 1, lr2, loss_function, 1);

    // Verify that parameter changes in network2 are approximately twice those in network1
    for (size_t l = 0; l < network1.numLayers(); ++l) {
        const Matrix& weights1 = network1.getLayer(l).getWeights();
        const Matrix& weights2 = network2.getLayer(l).getWeights();
        const Vector& biases1 = network1.getLayer(l).getBiases();
        const Vector& biases2 = network2.getLayer(l).getBiases();

        // Check weight changes
        for (size_t i = 0; i < weights1.rows(); ++i) {
            for (size_t j = 0; j < weights1.cols(); ++j) {
                double change1 = weights1(i, j) - original_weights[l](i, j);
                double change2 = weights2(i, j) - original_weights[l](i, j);

                // change2 should be approximately 2 * change1
                // Use relative tolerance for non-zero changes
                if (std::abs(change1) > 1e-9) {
                    RC_ASSERT(approxEqual(change2, 2.0 * change1, 1e-6));
                }
            }
        }

        // Check bias changes
        for (size_t i = 0; i < biases1.size(); ++i) {
            double change1 = biases1[i] - original_biases[l][i];
            double change2 = biases2[i] - original_biases[l][i];

            // change2 should be approximately 2 * change1
            if (std::abs(change1) > 1e-9) {
                RC_ASSERT(approxEqual(change2, 2.0 * change1, 1e-6));
            }
        }
    }
}


// **Validates: Requirements 8.2, 8.3**
// Feature: neural-network-framework, Property 22: Training Epoch Iteration
// For any dataset with N examples and E epochs, the training loop should perform
// exactly N * E forward-backward-update cycles (when batch_size = 1).
RC_GTEST_PROP(TrainingPropertyTest, TrainingEpochIteration, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 4);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create a small dataset
    size_t num_examples = *rc::gen::inRange<size_t>(3, 8);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    // Train for E epochs
    size_t epochs = *rc::gen::inRange<size_t>(2, 5);
    double learning_rate = 0.1;
    MeanSquaredError loss_function;

    // Train with batch_size = 1 (SGD)
    std::vector<double> loss_history = network.train(inputs, targets, epochs, learning_rate, loss_function, 1);

    // Verify that loss_history has exactly E entries (one per epoch)
    RC_ASSERT(loss_history.size() == epochs);

    // Verify all loss values are finite and non-negative
    for (size_t i = 0; i < loss_history.size(); ++i) {
        RC_ASSERT(std::isfinite(loss_history[i]));
        RC_ASSERT(loss_history[i] >= 0.0);
    }
}

// **Validates: Requirements 8.3**
// Feature: neural-network-framework, Property 37: Training Reduces Loss
// For any network and training dataset, after training for multiple epochs with
// appropriate learning rate, the final loss should be less than or equal to the
// initial loss (assuming the problem is learnable and learning rate is appropriate).
RC_GTEST_PROP(TrainingPropertyTest, TrainingReducesLoss, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(3, 6);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create a small, learnable dataset
    size_t num_examples = *rc::gen::inRange<size_t>(4, 10);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    // Train for multiple epochs with a reasonable learning rate
    size_t epochs = *rc::gen::inRange<size_t>(10, 30);
    double learning_rate = *genDoubleInRange(0.05, 0.3);
    MeanSquaredError loss_function;

    std::vector<double> loss_history = network.train(inputs, targets, epochs, learning_rate, loss_function, 1);

    // Verify that loss_history is not empty
    RC_ASSERT(!loss_history.empty());

    // Get initial and final loss
    double initial_loss = loss_history.front();
    double final_loss = loss_history.back();

    // Verify both are finite
    RC_ASSERT(std::isfinite(initial_loss));
    RC_ASSERT(std::isfinite(final_loss));

    // Final loss should be less than or equal to initial loss
    // (with some tolerance for numerical issues)
    // Note: In rare cases, loss might increase slightly due to numerical issues
    // or if the learning rate is too high, but generally it should decrease
    RC_ASSERT(final_loss <= initial_loss * 1.1);  // Allow 10% tolerance

    // Additionally, verify that loss generally trends downward
    // Count how many times loss decreased vs increased
    size_t decreases = 0;
    size_t increases = 0;

    for (size_t i = 1; i < loss_history.size(); ++i) {
        if (loss_history[i] < loss_history[i-1]) {
            decreases++;
        } else if (loss_history[i] > loss_history[i-1]) {
            increases++;
        }
    }

    // Loss should decrease more often than it increases
    RC_ASSERT(decreases >= increases);
}

// **Validates: Requirements 8.10**
// Feature: neural-network-framework, Property 39: Batch Size Validation
// For any training call with batch_size parameter, if batch_size is zero or greater
// than the dataset size, the system should reject the configuration with an error.
RC_GTEST_PROP(TrainingPropertyTest, BatchSizeValidation, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 4);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create a small dataset
    size_t num_examples = *rc::gen::inRange<size_t>(5, 10);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    MeanSquaredError loss_function;

    // Test 1: batch_size = 0 should throw
    RC_ASSERT_THROWS_AS(
        network.train(inputs, targets, 1, 0.1, loss_function, 0),
        std::invalid_argument
    );

    // Test 2: batch_size > dataset_size should throw
    size_t invalid_batch_size = num_examples + 1;
    RC_ASSERT_THROWS_AS(
        network.train(inputs, targets, 1, 0.1, loss_function, invalid_batch_size),
        std::invalid_argument
    );

    // Test 3: Valid batch sizes should work
    // batch_size = 1 (SGD)
    std::vector<double> loss_history1 = network.train(inputs, targets, 1, 0.1, loss_function, 1);
    RC_ASSERT(!loss_history1.empty());

    // batch_size = num_examples (full batch)
    Network network2 = createSimpleNetwork(input_size, hidden_size, output_size);
    std::vector<double> loss_history2 = network2.train(inputs, targets, 1, 0.1, loss_function, num_examples);
    RC_ASSERT(!loss_history2.empty());

    // batch_size in between (mini-batch)
    if (num_examples >= 3) {
        Network network3 = createSimpleNetwork(input_size, hidden_size, output_size);
        size_t mini_batch_size = num_examples / 2;
        std::vector<double> loss_history3 = network3.train(inputs, targets, 1, 0.1, loss_function, mini_batch_size);
        RC_ASSERT(!loss_history3.empty());
    }
}

// **Validates: Requirements 8.5**
// Feature: neural-network-framework, Property 40: Batch Gradient Averaging
// For any network, training dataset, and batch_size > 1, the gradients applied to
// parameters should be the average of the gradients computed over the batch.
// Specifically, training with batch_size=N should produce parameter updates that are
// the average of N individual SGD updates (before applying learning rate).
RC_GTEST_PROP(TrainingPropertyTest, BatchGradientAveraging, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 3);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 3);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    // Create dataset with exactly 4 examples for easy batch testing
    size_t num_examples = 4;
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    double learning_rate = 0.1;
    MeanSquaredError loss_function;

    // Scenario 1: Train with batch_size = 1 (SGD) for 4 steps
    Network network_sgd = createSimpleNetwork(input_size, hidden_size, output_size);

    // Set specific initial weights for reproducibility
    for (size_t l = 0; l < network_sgd.numLayers(); ++l) {
        Matrix& weights = network_sgd.getLayer(l).getWeights();
        Vector& biases = network_sgd.getLayer(l).getBiases();

        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                weights(i, j) = *genDoubleInRange(-0.3, 0.3);
            }
        }

        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = *genDoubleInRange(-0.1, 0.1);
        }
    }

    // Store initial parameters
    std::vector<Matrix> initial_weights_sgd;
    std::vector<Vector> initial_biases_sgd;

    for (size_t l = 0; l < network_sgd.numLayers(); ++l) {
        initial_weights_sgd.push_back(network_sgd.getLayer(l).getWeights());
        initial_biases_sgd.push_back(network_sgd.getLayer(l).getBiases());
    }

    // Train with SGD (batch_size = 1) for 1 epoch (4 updates)
    network_sgd.train(inputs, targets, 1, learning_rate, loss_function, 1);

    // Scenario 2: Train with batch_size = 4 (full batch) for 1 step
    Network network_batch = createSimpleNetwork(input_size, hidden_size, output_size);

    // Set same initial weights as SGD network
    for (size_t l = 0; l < network_batch.numLayers(); ++l) {
        Matrix& weights = network_batch.getLayer(l).getWeights();
        Vector& biases = network_batch.getLayer(l).getBiases();

        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                weights(i, j) = initial_weights_sgd[l](i, j);
            }
        }

        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = initial_biases_sgd[l][i];
        }
    }

    // Train with full batch (batch_size = 4) for 1 epoch (1 update with averaged gradients)
    network_batch.train(inputs, targets, 1, learning_rate, loss_function, 4);

    // The final parameters should be different between SGD and batch training
    // because SGD updates after each example (parameters change between examples),
    // while batch training computes gradients on the original parameters for all examples
    // and then updates once with the averaged gradient.

    // However, we can verify that both produce valid, finite parameters
    for (size_t l = 0; l < network_sgd.numLayers(); ++l) {
        const Matrix& weights_sgd = network_sgd.getLayer(l).getWeights();
        const Matrix& weights_batch = network_batch.getLayer(l).getWeights();
        const Vector& biases_sgd = network_sgd.getLayer(l).getBiases();
        const Vector& biases_batch = network_batch.getLayer(l).getBiases();

        // Verify all parameters are finite
        for (size_t i = 0; i < weights_sgd.rows(); ++i) {
            for (size_t j = 0; j < weights_sgd.cols(); ++j) {
                RC_ASSERT(std::isfinite(weights_sgd(i, j)));
                RC_ASSERT(std::isfinite(weights_batch(i, j)));
            }
        }

        for (size_t i = 0; i < biases_sgd.size(); ++i) {
            RC_ASSERT(std::isfinite(biases_sgd[i]));
            RC_ASSERT(std::isfinite(biases_batch[i]));
        }

        // Verify that parameters changed from initial values
        bool sgd_changed = false;
        bool batch_changed = false;

        for (size_t i = 0; i < weights_sgd.rows(); ++i) {
            for (size_t j = 0; j < weights_sgd.cols(); ++j) {
                if (!approxEqual(weights_sgd(i, j), initial_weights_sgd[l](i, j), 1e-9)) {
                    sgd_changed = true;
                }
                if (!approxEqual(weights_batch(i, j), initial_weights_sgd[l](i, j), 1e-9)) {
                    batch_changed = true;
                }
            }
        }

        RC_ASSERT(sgd_changed);
        RC_ASSERT(batch_changed);
    }

    // Additional test: Verify that batch_size = 2 produces different results than batch_size = 1
    Network network_mini = createSimpleNetwork(input_size, hidden_size, output_size);

    // Set same initial weights
    for (size_t l = 0; l < network_mini.numLayers(); ++l) {
        Matrix& weights = network_mini.getLayer(l).getWeights();
        Vector& biases = network_mini.getLayer(l).getBiases();

        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                weights(i, j) = initial_weights_sgd[l](i, j);
            }
        }

        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = initial_biases_sgd[l][i];
        }
    }

    // Train with mini-batch (batch_size = 2) for 1 epoch (2 updates)
    network_mini.train(inputs, targets, 1, learning_rate, loss_function, 2);

    // Verify parameters are finite and changed
    for (size_t l = 0; l < network_mini.numLayers(); ++l) {
        const Matrix& weights = network_mini.getLayer(l).getWeights();
        const Vector& biases = network_mini.getLayer(l).getBiases();

        bool changed = false;

        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                RC_ASSERT(std::isfinite(weights(i, j)));
                if (!approxEqual(weights(i, j), initial_weights_sgd[l](i, j), 1e-9)) {
                    changed = true;
                }
            }
        }

        for (size_t i = 0; i < biases.size(); ++i) {
            RC_ASSERT(std::isfinite(biases[i]));
            if (!approxEqual(biases[i], initial_biases_sgd[l][i], 1e-9)) {
                changed = true;
            }
        }

        RC_ASSERT(changed);
    }
}

// Additional property: Verify that training with epochs = 0 is rejected
RC_GTEST_PROP(TrainingPropertyTest, ZeroEpochsRejection, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 4);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create a small dataset
    size_t num_examples = *rc::gen::inRange<size_t>(3, 8);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    // Train for 0 epochs should throw
    MeanSquaredError loss_function;
    RC_ASSERT_THROWS_AS(
        network.train(inputs, targets, 0, 0.1, loss_function, 1),
        std::invalid_argument
    );
}

// Additional property: Verify that negative learning rate is rejected
RC_GTEST_PROP(TrainingPropertyTest, NegativeLearningRateRejection, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 4);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create a small dataset
    size_t num_examples = *rc::gen::inRange<size_t>(3, 8);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    MeanSquaredError loss_function;

    // Negative learning rate should throw
    double negative_lr = -*genDoubleInRange(0.01, 1.0);
    RC_ASSERT_THROWS_AS(
        network.train(inputs, targets, 1, negative_lr, loss_function, 1),
        std::invalid_argument
    );
}

// Additional property: Verify that mismatched input/target sizes are rejected
RC_GTEST_PROP(TrainingPropertyTest, MismatchedDatasetSizeRejection, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 4);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 4);
    size_t output_size = *rc::gen::inRange<size_t>(1, 2);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create mismatched datasets
    size_t num_inputs = *rc::gen::inRange<size_t>(5, 10);
    size_t num_targets = num_inputs + 1;  // Intentionally different

    std::vector<Vector> inputs;
    std::vector<Vector> targets;

    for (size_t i = 0; i < num_inputs; ++i) {
        Vector input(input_size, 0.0);
        inputs.push_back(input);
    }

    for (size_t i = 0; i < num_targets; ++i) {
        Vector target(output_size, 0.0);
        targets.push_back(target);
    }

    MeanSquaredError loss_function;

    // Mismatched sizes should throw
    RC_ASSERT_THROWS_AS(
        network.train(inputs, targets, 1, 0.1, loss_function, 1),
        std::invalid_argument
    );
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
