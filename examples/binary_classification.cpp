#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <random>
#include <vector>

/**
 * Binary Classification Example - Educational Neural Network Demonstration
 *
 * This example demonstrates binary classification on a linearly separable dataset.
 * Unlike XOR, this problem CAN be solved by a linear classifier, but we use a
 * simple neural network to demonstrate the framework's capabilities.
 *
 * Problem: Classify points as above or below a line
 * Decision boundary: y = 0.5 * x + 0.3
 * - Points above the line: class 1
 * - Points below the line: class 0
 *
 * This is a simpler problem than XOR because it is linearly separable.
 * A network should achieve >95% accuracy easily.
 *
 * Network Architecture:
 * - Input layer: 2 neurons (x, y coordinates)
 * - Hidden layer: 3 neurons (with tanh activation)
 * - Output layer: 1 neuron (with tanh activation)
 *
 * Requirements validated:
 * - 13.2: Binary classification example
 * - 13.4: Print training progress and results
 * - 13.5: Detailed educational comments
 */

/**
 * Generate a synthetic linearly separable dataset.
 *
 * Creates points in 2D space and labels them based on whether they are
 * above or below the line: y = 0.5 * x + 0.3
 *
 * @param num_samples Number of samples to generate
 * @param inputs Output vector for input points
 * @param targets Output vector for target labels
 * @param seed Random seed for reproducibility
 */
void generateLinearlySeparableDataset(
    size_t num_samples,
    std::vector<Vector>& inputs,
    std::vector<Vector>& targets,
    unsigned int seed = 42
) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Decision boundary: y = 0.5 * x + 0.3
    const double slope = 0.5;
    const double intercept = 0.3;

    inputs.clear();
    targets.clear();

    for (size_t i = 0; i < num_samples; ++i) {
        // Generate random point in [-1, 1] x [-1, 1]
        double x = dist(rng);
        double y = dist(rng);

        // Determine class based on position relative to line
        double line_y = slope * x + intercept;
        // Use -1 and 1 for tanh output (which ranges from -1 to 1)
        double label = (y > line_y) ? 1.0 : -1.0;

        inputs.push_back(Vector{x, y});
        targets.push_back(Vector{label});
    }
}

/**
 * Split dataset into training and test sets.
 *
 * @param inputs All input samples
 * @param targets All target labels
 * @param train_ratio Fraction of data to use for training (e.g., 0.8 for 80%)
 * @param train_inputs Output vector for training inputs
 * @param train_targets Output vector for training targets
 * @param test_inputs Output vector for test inputs
 * @param test_targets Output vector for test targets
 */
void splitDataset(
    const std::vector<Vector>& inputs,
    const std::vector<Vector>& targets,
    double train_ratio,
    std::vector<Vector>& train_inputs,
    std::vector<Vector>& train_targets,
    std::vector<Vector>& test_inputs,
    std::vector<Vector>& test_targets
) {
    size_t train_size = static_cast<size_t>(inputs.size() * train_ratio);

    train_inputs.clear();
    train_targets.clear();
    test_inputs.clear();
    test_targets.clear();

    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i < train_size) {
            train_inputs.push_back(inputs[i]);
            train_targets.push_back(targets[i]);
        } else {
            test_inputs.push_back(inputs[i]);
            test_targets.push_back(targets[i]);
        }
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Binary Classification - Neural Network Demo" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // ============================================================================
    // STEP 1: Generate linearly separable dataset
    // ============================================================================

    std::cout << "Step 1: Generating linearly separable dataset" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    const size_t total_samples = 200;
    const double train_ratio = 0.8;  // 80% training, 20% test

    std::vector<Vector> all_inputs;
    std::vector<Vector> all_targets;

    // Generate dataset with decision boundary: y = 0.5 * x + 0.3
    generateLinearlySeparableDataset(total_samples, all_inputs, all_targets, 42);

    std::cout << "Generated " << total_samples << " samples" << std::endl;
    std::cout << "Decision boundary: y = 0.5 * x + 0.3" << std::endl;
    std::cout << "  - Points above the line: class 1 (target = 1.0)" << std::endl;
    std::cout << "  - Points below the line: class 0 (target = -1.0)" << std::endl;
    std::cout << "Note: Using -1 and 1 as targets to match tanh output range" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 2: Split into training and test sets
    // ============================================================================

    std::cout << "Step 2: Splitting dataset" << std::endl;
    std::cout << "-------------------------" << std::endl;

    std::vector<Vector> train_inputs, train_targets;
    std::vector<Vector> test_inputs, test_targets;

    splitDataset(all_inputs, all_targets, train_ratio,
                 train_inputs, train_targets,
                 test_inputs, test_targets);

    std::cout << "Training set: " << train_inputs.size() << " samples" << std::endl;
    std::cout << "Test set: " << test_inputs.size() << " samples" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 3: Create the neural network architecture
    // ============================================================================

    std::cout << "Step 3: Creating network architecture" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // Create a network with architecture: [2, 3, 1]
    // - 2 input neurons (x, y coordinates)
    // - 3 hidden neurons (simple hidden layer)
    // - 1 output neuron (binary classification)
    Network network;

    // Use tanh activation function for all layers
    // Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    // Output range: (-1, 1), zero-centered which often works well
    auto tanh_activation = std::make_shared<Tanh>();

    // Add hidden layer: 2 inputs -> 3 outputs
    network.addLayer(2, 3, tanh_activation);

    // Add output layer: 3 inputs -> 1 output
    network.addLayer(3, 1, tanh_activation);

    std::cout << "Network architecture: [2, 3, 1]" << std::endl;
    std::cout << "  - Input layer: 2 neurons (x, y coordinates)" << std::endl;
    std::cout << "  - Hidden layer: 3 neurons (tanh activation)" << std::endl;
    std::cout << "  - Output layer: 1 neuron (tanh activation)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 4: Initialize weights with a fixed seed for reproducibility
    // ============================================================================

    std::cout << "Step 4: Initializing weights" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // Set a fixed random seed to ensure reproducible results
    Matrix::setSeed(42);

    // Initialize weights using Xavier initialization
    // Xavier initialization is appropriate for tanh activation
    network.getLayer(0).initializeXavier(2, 3);  // Hidden layer: 2 inputs, 3 outputs
    network.getLayer(1).initializeXavier(3, 1);  // Output layer: 3 inputs, 1 output

    std::cout << "Weights initialized with Xavier initialization (seed = 42)" << std::endl;
    std::cout << "Xavier initialization is optimal for tanh activation" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 5: Set up training parameters
    // ============================================================================

    std::cout << "Step 5: Configuring training parameters" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    const size_t epochs = 1000;            // Number of complete passes through the dataset
    const double learning_rate = 0.1;      // Step size for gradient descent
    const size_t print_interval = 100;     // Print progress every N epochs

    std::cout << "Training configuration:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Learning rate: " << learning_rate << std::endl;
    std::cout << "  - Loss function: Mean Squared Error (MSE)" << std::endl;
    std::cout << "  - Optimization: Stochastic Gradient Descent (SGD)" << std::endl;
    std::cout << "  - Progress interval: every " << print_interval << " epochs" << std::endl;
    std::cout << std::endl;

    // Create loss function
    // MSE = (1/n) * Σ(predicted - target)²
    MeanSquaredError loss_function;

    // Create training monitor to track progress
    TrainingMonitor monitor(100);

    // ============================================================================
    // STEP 6: Train the network
    // ============================================================================

    std::cout << "Step 6: Training the network" << std::endl;
    std::cout << "----------------------------" << std::endl;
    std::cout << "Training in progress..." << std::endl;
    std::cout << std::endl;

    // Train the network using backpropagation
    std::vector<double> loss_history = network.train(
        train_inputs,
        train_targets,
        epochs,
        learning_rate,
        loss_function,
        1,  // batch_size = 1 (Stochastic Gradient Descent)
        &monitor
    );

    // Print training progress at regular intervals
    std::cout << "Epoch | Average Loss" << std::endl;
    std::cout << "------|-------------" << std::endl;
    for (size_t epoch = 0; epoch < epochs; epoch += print_interval) {
        if (epoch < loss_history.size()) {
            std::cout << std::setw(5) << epoch << " | "
                      << std::fixed << std::setprecision(6)
                      << loss_history[epoch] << std::endl;
        }
    }
    // Print final epoch
    std::cout << std::setw(5) << epochs << " | "
              << std::fixed << std::setprecision(6)
              << loss_history.back() << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 7: Evaluate on training set
    // ============================================================================

    std::cout << "Step 7: Evaluating on training set" << std::endl;
    std::cout << "-----------------------------------" << std::endl;

    int train_correct = 0;
    const double threshold = 0.0;  // For tanh output: >0 is class 1, <=0 is class 0

    for (size_t i = 0; i < train_inputs.size(); ++i) {
        Vector prediction = network.predict(train_inputs[i]);

        // For tanh activation, output is in range (-1, 1)
        // Map to binary: >0 -> class 1 (target 1.0), <=0 -> class 0 (target -1.0)
        double predicted_class = (prediction[0] > threshold) ? 1.0 : -1.0;
        bool is_correct = (predicted_class == train_targets[i][0]);

        if (is_correct) {
            train_correct++;
        }
    }

    double train_accuracy = (double)train_correct / train_inputs.size() * 100.0;

    std::cout << "Training accuracy: " << train_correct << "/" << train_inputs.size()
              << " = " << std::fixed << std::setprecision(1) << train_accuracy << "%" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 8: Evaluate on test set
    // ============================================================================

    std::cout << "Step 8: Evaluating on test set" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    int test_correct = 0;

    for (size_t i = 0; i < test_inputs.size(); ++i) {
        Vector prediction = network.predict(test_inputs[i]);

        // For tanh activation, output is in range (-1, 1)
        // Map to binary: >0 -> class 1 (target 1.0), <=0 -> class 0 (target -1.0)
        double predicted_class = (prediction[0] > threshold) ? 1.0 : -1.0;
        bool is_correct = (predicted_class == test_targets[i][0]);

        if (is_correct) {
            test_correct++;
        }
    }

    double test_accuracy = (double)test_correct / test_inputs.size() * 100.0;

    std::cout << "Test accuracy: " << test_correct << "/" << test_inputs.size()
              << " = " << std::fixed << std::setprecision(1) << test_accuracy << "%" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 9: Show some example predictions
    // ============================================================================

    std::cout << "Step 9: Example predictions on test set" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    std::cout << "   X    |   Y    | Target | Prediction | Correct?" << std::endl;
    std::cout << "--------|--------|--------|------------|----------" << std::endl;

    // Show first 10 test examples
    size_t num_examples = std::min(size_t(10), test_inputs.size());
    for (size_t i = 0; i < num_examples; ++i) {
        Vector prediction = network.predict(test_inputs[i]);
        double predicted_class = (prediction[0] > threshold) ? 1.0 : -1.0;
        bool is_correct = (predicted_class == test_targets[i][0]);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(7) << test_inputs[i][0] << " | "
                  << std::setw(6) << test_inputs[i][1] << " | "
                  << std::setw(6) << test_targets[i][0] << " | "
                  << std::setw(10) << prediction[0] << " | "
                  << (is_correct ? "✓" : "✗") << std::endl;
    }

    std::cout << std::endl;

    // ============================================================================
    // STEP 10: Final results summary
    // ============================================================================

    std::cout << "Step 10: Final results" << std::endl;
    std::cout << "----------------------" << std::endl;

    std::cout << "Training accuracy: " << std::fixed << std::setprecision(1)
              << train_accuracy << "%" << std::endl;
    std::cout << "Test accuracy: " << std::fixed << std::setprecision(1)
              << test_accuracy << "%" << std::endl;
    std::cout << "Final training loss: " << std::fixed << std::setprecision(6)
              << loss_history.back() << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // Educational Summary
    // ============================================================================

    std::cout << "========================================" << std::endl;
    std::cout << "  Educational Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Key Concepts Demonstrated:" << std::endl;
    std::cout << "1. Linear Separability:" << std::endl;
    std::cout << "   This problem is linearly separable, meaning a straight line" << std::endl;
    std::cout << "   can perfectly separate the two classes. Even a simple network" << std::endl;
    std::cout << "   can learn this pattern easily." << std::endl;
    std::cout << std::endl;

    std::cout << "2. Train/Test Split:" << std::endl;
    std::cout << "   We split the data into training (" << train_ratio * 100 << "%) and test ("
              << (1 - train_ratio) * 100 << "%) sets." << std::endl;
    std::cout << "   This allows us to evaluate how well the network generalizes to" << std::endl;
    std::cout << "   unseen data, not just memorize the training examples." << std::endl;
    std::cout << std::endl;

    std::cout << "3. Tanh Activation:" << std::endl;
    std::cout << "   Tanh outputs values in range (-1, 1), which is zero-centered." << std::endl;
    std::cout << "   This often leads to faster convergence compared to sigmoid." << std::endl;
    std::cout << "   We use targets -1 and 1 to match tanh's output range." << std::endl;
    std::cout << "   Classification: >0 is class 1, <=0 is class 0." << std::endl;
    std::cout << std::endl;

    std::cout << "4. Generalization:" << std::endl;
    std::cout << "   The test accuracy shows how well the network generalizes." << std::endl;
    std::cout << "   Good generalization means the network learned the underlying" << std::endl;
    std::cout << "   pattern, not just memorized the training data." << std::endl;
    std::cout << std::endl;

    if (test_accuracy >= 95.0) {
        std::cout << "✓ SUCCESS: The network achieved excellent test accuracy!" << std::endl;
        std::cout << "  With " << test_accuracy << "% accuracy on unseen data, the network" << std::endl;
        std::cout << "  successfully learned the linear decision boundary." << std::endl;
    } else if (test_accuracy >= 85.0) {
        std::cout << "✓ GOOD: The network achieved good test accuracy!" << std::endl;
        std::cout << "  With " << test_accuracy << "% accuracy, the network learned the pattern" << std::endl;
        std::cout << "  reasonably well. Some misclassifications may occur near the boundary." << std::endl;
    } else {
        std::cout << "⚠ NEEDS IMPROVEMENT: Test accuracy is " << test_accuracy << "%." << std::endl;
        std::cout << "  This may indicate:" << std::endl;
        std::cout << "  - More training epochs needed" << std::endl;
        std::cout << "  - Different learning rate required" << std::endl;
        std::cout << "  - Different network architecture" << std::endl;
        std::cout << "  Try adjusting hyperparameters and running again." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
