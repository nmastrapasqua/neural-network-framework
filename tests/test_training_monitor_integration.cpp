#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include <iostream>
#include <memory>

/**
 * Test: TrainingMonitor Integration with Network Training
 *
 * This test verifies that:
 * 1. Network.train() accepts an optional TrainingMonitor pointer
 * 2. When provided, the monitor's recordEpoch() is called at the end of each epoch
 * 3. When provided, the monitor's printProgress() is called at the end of each epoch
 * 4. The monitor correctly records loss and accuracy history
 *
 * Validates Requirements 8.4, 8.5, 8.6
 */

int main() {
    std::cout << "Test: TrainingMonitor Integration\n";
    std::cout << "==================================\n\n";

    // Test 1: Train without monitor (backward compatibility)
    {
        std::cout << "Test 1: Train without monitor (backward compatibility)\n";

        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        // Initialize weights
        network.getLayer(0).initializeXavier(2, 4);
        network.getLayer(1).initializeXavier(4, 1);

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

        // Train without monitor (nullptr is default)
        auto loss_history = network.train(inputs, targets, 5, 0.5, loss_fn, 1);

        if (loss_history.size() == 5) {
            std::cout << "  PASS: Training completed without monitor\n";
            std::cout << "  Loss history size: " << loss_history.size() << "\n";
        } else {
            std::cout << "  FAIL: Expected 5 epochs, got " << loss_history.size() << "\n";
            return 1;
        }

        std::cout << "\n";
    }

    // Test 2: Train with monitor
    {
        std::cout << "Test 2: Train with monitor\n";

        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        // Initialize weights
        network.getLayer(0).initializeXavier(2, 4);
        network.getLayer(1).initializeXavier(4, 1);

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
        TrainingMonitor monitor;

        // Train with monitor
        std::cout << "  Training with monitor (progress will be printed):\n";
        auto loss_history = network.train(inputs, targets, 5, 0.5, loss_fn, 1, &monitor);

        // Verify monitor recorded all epochs
        const auto& recorded_loss_history = monitor.getLossHistory();
        const auto& accuracy_history = monitor.getAccuracyHistory();

        if (recorded_loss_history.size() == 5) {
            std::cout << "  PASS: Monitor recorded 5 epochs\n";
        } else {
            std::cout << "  FAIL: Monitor recorded " << recorded_loss_history.size() << " epochs, expected 5\n";
            return 1;
        }

        if (accuracy_history.size() == 5) {
            std::cout << "  PASS: Monitor recorded 5 accuracy values\n";
        } else {
            std::cout << "  FAIL: Monitor recorded " << accuracy_history.size() << " accuracy values, expected 5\n";
            return 1;
        }

        // Verify loss history matches between network and monitor
        bool loss_matches = true;
        for (size_t i = 0; i < 5; ++i) {
            if (loss_history[i] != recorded_loss_history[i]) {
                loss_matches = false;
                break;
            }
        }

        if (loss_matches) {
            std::cout << "  PASS: Loss history matches between network and monitor\n";
        } else {
            std::cout << "  FAIL: Loss history mismatch\n";
            return 1;
        }

        // Verify average loss calculation
        double avg_loss = monitor.getAverageLoss();
        double expected_avg = 0.0;
        for (double loss : recorded_loss_history) {
            expected_avg += loss;
        }
        expected_avg /= recorded_loss_history.size();

        if (std::abs(avg_loss - expected_avg) < 1e-9) {
            std::cout << "  PASS: Average loss calculation correct\n";
        } else {
            std::cout << "  FAIL: Average loss mismatch\n";
            return 1;
        }

        std::cout << "\n";
        std::cout << "  Final metrics:\n";
        std::cout << "    Average loss: " << avg_loss << "\n";
        std::cout << "    Final accuracy: " << accuracy_history.back() << "\n";

        std::cout << "\n";
    }

    // Test 3: Train with monitor and batch_size > 1
    {
        std::cout << "Test 3: Train with monitor and batch_size = 2\n";

        Network network;
        network.addLayer(2, 4, std::make_shared<Sigmoid>());
        network.addLayer(4, 1, std::make_shared<Sigmoid>());

        // Initialize weights
        network.getLayer(0).initializeXavier(2, 4);
        network.getLayer(1).initializeXavier(4, 1);

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
        TrainingMonitor monitor;

        // Train with monitor and batch_size = 2
        std::cout << "  Training with monitor and batch_size = 2:\n";
        auto loss_history = network.train(inputs, targets, 3, 0.5, loss_fn, 2, &monitor);

        // Verify monitor recorded all epochs
        const auto& recorded_loss_history = monitor.getLossHistory();

        if (recorded_loss_history.size() == 3) {
            std::cout << "  PASS: Monitor recorded 3 epochs with batch training\n";
        } else {
            std::cout << "  FAIL: Monitor recorded " << recorded_loss_history.size() << " epochs, expected 3\n";
            return 1;
        }

        std::cout << "\n";
    }

    std::cout << "All tests passed!\n";
    return 0;
}
