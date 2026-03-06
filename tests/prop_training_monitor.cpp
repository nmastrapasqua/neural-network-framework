#include "training_monitor.h"
#include "network.h"
#include "activation.h"
#include "loss.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <memory>
#include <cmath>

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

// **Validates: Requirements 8.4, 8.5**
// Feature: neural-network-framework, Property 23: Loss History Recording
// For any training session with E epochs, the loss history should contain exactly E entries,
// one per epoch.
RC_GTEST_PROP(TrainingMonitorPropertyTest, LossHistoryRecording, ()) {
    // Create a small network
    size_t input_size = *rc::gen::inRange<size_t>(2, 5);
    size_t hidden_size = *rc::gen::inRange<size_t>(2, 5);
    size_t output_size = *rc::gen::inRange<size_t>(1, 3);

    Network network = createSimpleNetwork(input_size, hidden_size, output_size);

    // Create a small dataset
    size_t num_examples = *rc::gen::inRange<size_t>(3, 10);
    std::vector<Vector> inputs;
    std::vector<Vector> targets;
    createSimpleDataset(num_examples, input_size, output_size, inputs, targets);

    // Generate random number of epochs E
    size_t epochs = *rc::gen::inRange<size_t>(1, 20);

    // Create TrainingMonitor
    TrainingMonitor monitor;

    // Train the network with the monitor
    double learning_rate = 0.1;
    MeanSquaredError loss_function;
    size_t batch_size = *rc::gen::inRange<size_t>(1, num_examples + 1);

    network.train(inputs, targets, epochs, learning_rate, loss_function, batch_size, &monitor);

    // Property: Loss history should contain exactly E entries
    const std::vector<double>& loss_history = monitor.getLossHistory();
    RC_ASSERT(loss_history.size() == epochs);

    // Additional verification: All loss values should be finite and non-negative
    for (size_t i = 0; i < loss_history.size(); ++i) {
        RC_ASSERT(std::isfinite(loss_history[i]));
        RC_ASSERT(loss_history[i] >= 0.0);
    }

    // Verify accuracy history also has E entries
    const std::vector<double>& accuracy_history = monitor.getAccuracyHistory();
    RC_ASSERT(accuracy_history.size() == epochs);

    // All accuracy values should be finite and in range [0, 1]
    for (size_t i = 0; i < accuracy_history.size(); ++i) {
        RC_ASSERT(std::isfinite(accuracy_history[i]));
        RC_ASSERT(accuracy_history[i] >= 0.0);
        RC_ASSERT(accuracy_history[i] <= 1.0);
    }
}

// Additional property: Verify that TrainingMonitor correctly records each epoch
RC_GTEST_PROP(TrainingMonitorPropertyTest, EpochRecordingCorrectness, ()) {
    TrainingMonitor monitor;

    // Generate random number of epochs
    size_t num_epochs = *rc::gen::inRange<size_t>(1, 50);

    // Record random loss and accuracy values for each epoch
    std::vector<double> expected_losses;
    std::vector<double> expected_accuracies;

    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Generate loss in range [0.0, 10.0]
        double loss = *rc::gen::scale(0.1, rc::gen::inRange<int>(0, 100));
        // Generate accuracy in range [0.0, 1.0]
        double accuracy = *rc::gen::scale(0.01, rc::gen::inRange<int>(0, 100));

        expected_losses.push_back(loss);
        expected_accuracies.push_back(accuracy);

        monitor.recordEpoch(epoch, loss, accuracy);
    }

    // Verify that loss history matches what we recorded
    const std::vector<double>& loss_history = monitor.getLossHistory();
    RC_ASSERT(loss_history.size() == num_epochs);

    for (size_t i = 0; i < num_epochs; ++i) {
        RC_ASSERT(loss_history[i] == expected_losses[i]);
    }

    // Verify that accuracy history matches what we recorded
    const std::vector<double>& accuracy_history = monitor.getAccuracyHistory();
    RC_ASSERT(accuracy_history.size() == num_epochs);

    for (size_t i = 0; i < num_epochs; ++i) {
        RC_ASSERT(accuracy_history[i] == expected_accuracies[i]);
    }
}

// Additional property: Verify average loss calculation
RC_GTEST_PROP(TrainingMonitorPropertyTest, AverageLossCalculation, ()) {
    TrainingMonitor monitor;

    // Generate random number of epochs
    size_t num_epochs = *rc::gen::inRange<size_t>(1, 30);

    // Record random loss values
    double sum = 0.0;
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) {
        // Generate loss in range [0.0, 5.0]
        double loss = *rc::gen::scale(0.05, rc::gen::inRange<int>(0, 100));
        // Generate accuracy in range [0.0, 1.0]
        double accuracy = *rc::gen::scale(0.01, rc::gen::inRange<int>(0, 100));

        sum += loss;
        monitor.recordEpoch(epoch, loss, accuracy);
    }

    // Calculate expected average
    double expected_average = sum / num_epochs;

    // Verify that getAverageLoss() returns the correct average
    double actual_average = monitor.getAverageLoss();

    // Use relative tolerance for floating point comparison
    double tolerance = 1e-9;
    RC_ASSERT(std::abs(actual_average - expected_average) < tolerance);
}

// Additional property: Empty monitor should return 0.0 for average loss
RC_GTEST_PROP(TrainingMonitorPropertyTest, EmptyMonitorAverageLoss, ()) {
    TrainingMonitor monitor;

    // Empty monitor should return 0.0 for average loss
    RC_ASSERT(monitor.getAverageLoss() == 0.0);

    // Loss history should be empty
    RC_ASSERT(monitor.getLossHistory().empty());

    // Accuracy history should be empty
    RC_ASSERT(monitor.getAccuracyHistory().empty());
}

// Additional property: Verify that multiple training sessions accumulate correctly
RC_GTEST_PROP(TrainingMonitorPropertyTest, MultipleTrainingSessions, ()) {
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

    // Create TrainingMonitor
    TrainingMonitor monitor;

    // First training session
    size_t epochs1 = *rc::gen::inRange<size_t>(1, 10);
    double learning_rate = 0.1;
    MeanSquaredError loss_function;

    network.train(inputs, targets, epochs1, learning_rate, loss_function, 1, &monitor);

    // Verify first session recorded correctly
    RC_ASSERT(monitor.getLossHistory().size() == epochs1);

    // Second training session (continuing with same monitor)
    size_t epochs2 = *rc::gen::inRange<size_t>(1, 10);
    network.train(inputs, targets, epochs2, learning_rate, loss_function, 1, &monitor);

    // Verify that both sessions are recorded (accumulated)
    size_t total_epochs = epochs1 + epochs2;
    RC_ASSERT(monitor.getLossHistory().size() == total_epochs);
    RC_ASSERT(monitor.getAccuracyHistory().size() == total_epochs);
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
