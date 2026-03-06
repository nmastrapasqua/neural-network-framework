#include "network.h"
#include "activation.h"
#include "loss.h"
#include <iostream>
#include <memory>
#include <cmath>

// Simple test to verify train() method works correctly
int main() {
    std::cout << "Testing Network::train() method with batch support...\n\n";

    // Test 1: Simple XOR-like problem with batch_size = 1 (SGD)
    std::cout << "Test 1: Training with batch_size = 1 (SGD)\n";
    {
        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        // Initialize weights
        network.getLayer(0).initializeWeights(-0.5, 0.5);
        network.getLayer(1).initializeWeights(-0.5, 0.5);

        // XOR dataset
        std::vector<Vector> inputs = {
            Vector({0.0, 0.0}),
            Vector({0.0, 1.0}),
            Vector({1.0, 0.0}),
            Vector({1.0, 1.0})
        };

        std::vector<Vector> targets = {
            Vector({0.0}),
            Vector({1.0}),
            Vector({1.0}),
            Vector({0.0})
        };

        MeanSquaredError loss_fn;

        // Train with batch_size = 1 (SGD)
        auto loss_history = network.train(inputs, targets, 10, 0.5, loss_fn, 1);

        std::cout << "  Initial loss: " << loss_history[0] << "\n";
        std::cout << "  Final loss: " << loss_history.back() << "\n";
        std::cout << "  Loss should decrease: " << (loss_history.back() < loss_history[0] ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 2: Training with batch_size = 2 (Mini-batch)
    std::cout << "Test 2: Training with batch_size = 2 (Mini-batch)\n";
    {
        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        // Initialize weights
        network.getLayer(0).initializeWeights(-0.5, 0.5);
        network.getLayer(1).initializeWeights(-0.5, 0.5);

        // XOR dataset
        std::vector<Vector> inputs = {
            Vector({0.0, 0.0}),
            Vector({0.0, 1.0}),
            Vector({1.0, 0.0}),
            Vector({1.0, 1.0})
        };

        std::vector<Vector> targets = {
            Vector({0.0}),
            Vector({1.0}),
            Vector({1.0}),
            Vector({0.0})
        };

        MeanSquaredError loss_fn;

        // Train with batch_size = 2
        auto loss_history = network.train(inputs, targets, 10, 0.5, loss_fn, 2);

        std::cout << "  Initial loss: " << loss_history[0] << "\n";
        std::cout << "  Final loss: " << loss_history.back() << "\n";
        std::cout << "  Loss should decrease: " << (loss_history.back() < loss_history[0] ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 3: Training with batch_size = dataset_size (Batch GD)
    std::cout << "Test 3: Training with batch_size = 4 (Batch Gradient Descent)\n";
    {
        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        // Initialize weights
        network.getLayer(0).initializeWeights(-0.5, 0.5);
        network.getLayer(1).initializeWeights(-0.5, 0.5);

        // XOR dataset
        std::vector<Vector> inputs = {
            Vector({0.0, 0.0}),
            Vector({0.0, 1.0}),
            Vector({1.0, 0.0}),
            Vector({1.0, 1.0})
        };

        std::vector<Vector> targets = {
            Vector({0.0}),
            Vector({1.0}),
            Vector({1.0}),
            Vector({0.0})
        };

        MeanSquaredError loss_fn;

        // Train with batch_size = 4 (full batch)
        auto loss_history = network.train(inputs, targets, 10, 0.5, loss_fn, 4);

        std::cout << "  Initial loss: " << loss_history[0] << "\n";
        std::cout << "  Final loss: " << loss_history.back() << "\n";
        std::cout << "  Loss should decrease: " << (loss_history.back() < loss_history[0] ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 4: Validation tests
    std::cout << "Test 4: Validation tests\n";
    {
        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        std::vector<Vector> inputs = {Vector({0.0, 0.0})};
        std::vector<Vector> targets = {Vector({0.0})};
        MeanSquaredError loss_fn;

        // Test batch_size = 0 (should throw)
        try {
            network.train(inputs, targets, 10, 0.5, loss_fn, 0);
            std::cout << "  batch_size = 0: FAIL (should throw)\n";
        } catch (const std::invalid_argument& e) {
            std::cout << "  batch_size = 0: PASS (correctly rejected)\n";
        }

        // Test batch_size > dataset_size (should throw)
        try {
            network.train(inputs, targets, 10, 0.5, loss_fn, 2);
            std::cout << "  batch_size > dataset_size: FAIL (should throw)\n";
        } catch (const std::invalid_argument& e) {
            std::cout << "  batch_size > dataset_size: PASS (correctly rejected)\n";
        }

        // Test epochs = 0 (should throw)
        try {
            network.train(inputs, targets, 0, 0.5, loss_fn, 1);
            std::cout << "  epochs = 0: FAIL (should throw)\n";
        } catch (const std::invalid_argument& e) {
            std::cout << "  epochs = 0: PASS (correctly rejected)\n";
        }

        // Test learning_rate <= 0 (should throw)
        try {
            network.train(inputs, targets, 10, 0.0, loss_fn, 1);
            std::cout << "  learning_rate = 0: FAIL (should throw)\n";
        } catch (const std::invalid_argument& e) {
            std::cout << "  learning_rate = 0: PASS (correctly rejected)\n";
        }

        // Test empty inputs (should throw)
        try {
            std::vector<Vector> empty_inputs;
            std::vector<Vector> empty_targets;
            network.train(empty_inputs, empty_targets, 10, 0.5, loss_fn, 1);
            std::cout << "  empty dataset: FAIL (should throw)\n";
        } catch (const std::invalid_argument& e) {
            std::cout << "  empty dataset: PASS (correctly rejected)\n";
        }

        // Test mismatched inputs/targets size (should throw)
        try {
            std::vector<Vector> inputs2 = {Vector({0.0, 0.0}), Vector({1.0, 1.0})};
            std::vector<Vector> targets2 = {Vector({0.0})};
            network.train(inputs2, targets2, 10, 0.5, loss_fn, 1);
            std::cout << "  mismatched inputs/targets: FAIL (should throw)\n";
        } catch (const std::invalid_argument& e) {
            std::cout << "  mismatched inputs/targets: PASS (correctly rejected)\n";
        }
    }

    std::cout << "\nAll tests completed!\n";

    return 0;
}
