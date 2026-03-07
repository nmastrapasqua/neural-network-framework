#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <random>

/**
 * XOR Problem Example - Educational Neural Network Demonstration
 *
 * This example demonstrates that a simple feedforward neural network can learn
 * the XOR (exclusive OR) function, which is a classic problem in neural network
 * history because it is NOT linearly separable.
 *
 * XOR Truth Table:
 *   Input 1 | Input 2 | Output
 *   --------|---------|-------
 *      0    |    0    |   0
 *      0    |    1    |   1
 *      1    |    0    |   1
 *      1    |    1    |   0
 *
 * Historical Context:
 * In 1969, Marvin Minsky and Seymour Papert showed that a single-layer perceptron
 * (linear classifier) cannot solve XOR. This limitation was a major setback for
 * neural network research. However, a multi-layer network with a hidden layer
 * CAN solve XOR, which was later demonstrated with the backpropagation algorithm.
 *
 * Network Architecture:
 * - Input layer: 2 neurons (for the two binary inputs)
 * - Hidden layer: 4 neurons (with sigmoid activation)
 * - Output layer: 1 neuron (with sigmoid activation)
 *
 * The hidden layer allows the network to learn non-linear decision boundaries,
 * which is essential for solving XOR.
 *
 * Requirements validated:
 * - 13.1: XOR problem example with training and testing
 * - 13.4: Print training progress at regular intervals
 * - 13.5: Detailed educational comments explaining the implementation
 */

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  XOR Problem - Neural Network Demo" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // ============================================================================
    // STEP 1: Define the XOR dataset
    // ============================================================================

    std::cout << "Step 1: Defining XOR dataset" << std::endl;
    std::cout << "----------------------------" << std::endl;

    // XOR is a binary classification problem with 4 possible input combinations
    std::vector<Vector> inputs = {
        Vector{0.0, 0.0},  // Input: [0, 0] -> Expected output: 0
        Vector{0.0, 1.0},  // Input: [0, 1] -> Expected output: 1
        Vector{1.0, 0.0},  // Input: [1, 0] -> Expected output: 1
        Vector{1.0, 1.0}   // Input: [1, 1] -> Expected output: 0
    };

    std::vector<Vector> targets = {
        Vector{0.0},  // Target for [0, 0]
        Vector{1.0},  // Target for [0, 1]
        Vector{1.0},  // Target for [1, 0]
        Vector{0.0}   // Target for [1, 1]
    };

    std::cout << "XOR Truth Table:" << std::endl;
    std::cout << "  Input 1 | Input 2 | Target Output" << std::endl;
    std::cout << "  --------|---------|---------------" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::cout << "     " << inputs[i][0] << "    |    "
                  << inputs[i][1] << "    |      "
                  << targets[i][0] << std::endl;
    }
    std::cout << std::endl;

    // ============================================================================
    // STEP 2: Create the neural network architecture
    // ============================================================================

    std::cout << "Step 2: Creating network architecture" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // Create a network with architecture: [2, 4, 1]
    // - 2 input neurons (for the two XOR inputs)
    // - 4 hidden neurons (enough to learn the XOR pattern)
    // - 1 output neuron (for the binary classification result)
    Network network;

    // Use sigmoid activation function for all layers
    // Sigmoid: σ(x) = 1 / (1 + e^(-x))
    // Output range: (0, 1), which is perfect for binary classification
    auto sigmoid = std::make_shared<Sigmoid>();

    // Add hidden layer: 2 inputs -> 4 outputs
    network.addLayer(2, 4, sigmoid);

    // Add output layer: 4 inputs -> 1 output
    network.addLayer(4, 1, sigmoid);

    std::cout << "Network architecture: [2, 4, 1]" << std::endl;
    std::cout << "  - Input layer: 2 neurons" << std::endl;
    std::cout << "  - Hidden layer: 4 neurons (sigmoid activation)" << std::endl;
    std::cout << "  - Output layer: 1 neuron (sigmoid activation)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 3: Initialize weights with a fixed seed for reproducibility
    // ============================================================================

    std::cout << "Step 3: Initializing weights" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // Set a fixed random seed to ensure reproducible results
    // This is important for educational purposes and debugging
    std::srand(42);

    // Initialize weights using Xavier initialization
    // Xavier initialization: weights ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
    // This initialization helps prevent vanishing/exploding gradients
    network.getLayer(0).initializeXavier(2, 4);  // Hidden layer: 2 inputs, 4 outputs
    network.getLayer(1).initializeXavier(4, 1);  // Output layer: 4 inputs, 1 output

    std::cout << "Weights initialized with Xavier initialization (seed = 42)" << std::endl;
    std::cout << "Xavier initialization helps prevent vanishing/exploding gradients" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 4: Set up training parameters
    // ============================================================================

    std::cout << "Step 4: Configuring training parameters" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    const size_t epochs = 10000;           // Number of complete passes through the dataset
    const double learning_rate = 0.5;      // Step size for gradient descent
    const size_t print_interval = 1000;    // Print progress every N epochs

    std::cout << "Training configuration:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Learning rate: " << learning_rate << std::endl;
    std::cout << "  - Loss function: Mean Squared Error (MSE)" << std::endl;
    std::cout << "  - Optimization: Stochastic Gradient Descent (SGD)" << std::endl;
    std::cout << "  - Progress interval: every " << print_interval << " epochs" << std::endl;
    std::cout << std::endl;

    // Create loss function
    // MSE = (1/n) * Σ(predicted - target)²
    // MSE is commonly used for regression and binary classification
    MeanSquaredError loss_function;

    // Create training monitor to track progress
    TrainingMonitor monitor;

    // ============================================================================
    // STEP 5: Train the network
    // ============================================================================

    std::cout << "Step 5: Training the network" << std::endl;
    std::cout << "----------------------------" << std::endl;
    std::cout << "Training in progress..." << std::endl;
    std::cout << std::endl;

    // Train the network using backpropagation
    // The training loop will:
    // 1. Forward pass: compute predictions for each input
    // 2. Compute loss: measure error between prediction and target
    // 3. Backward pass: compute gradients using backpropagation
    // 4. Update weights: apply gradient descent to minimize loss
    std::vector<double> loss_history = network.train(
        inputs,
        targets,
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
    // STEP 6: Test the trained network
    // ============================================================================

    std::cout << "Step 6: Testing the trained network" << std::endl;
    std::cout << "------------------------------------" << std::endl;

    std::cout << "Predictions on XOR inputs:" << std::endl;
    std::cout << "  Input 1 | Input 2 | Target | Prediction | Correct?" << std::endl;
    std::cout << "  --------|---------|--------|------------|----------" << std::endl;

    int correct_predictions = 0;
    const double threshold = 0.5;  // Classification threshold

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Make prediction
        Vector prediction = network.predict(inputs[i]);

        // Determine if prediction is correct
        // For binary classification: round to 0 or 1 using threshold
        double predicted_class = (prediction[0] >= threshold) ? 1.0 : 0.0;
        bool is_correct = (predicted_class == targets[i][0]);

        if (is_correct) {
            correct_predictions++;
        }

        // Print results
        std::cout << "     " << inputs[i][0] << "    |    "
                  << inputs[i][1] << "    |   "
                  << targets[i][0] << "    |   "
                  << std::fixed << std::setprecision(4) << prediction[0] << "    | "
                  << (is_correct ? "✓" : "✗") << std::endl;
    }

    std::cout << std::endl;

    // ============================================================================
    // STEP 7: Calculate and display accuracy
    // ============================================================================

    std::cout << "Step 7: Final results" << std::endl;
    std::cout << "---------------------" << std::endl;

    double accuracy = (double)correct_predictions / inputs.size() * 100.0;

    std::cout << "Accuracy: " << correct_predictions << "/" << inputs.size()
              << " = " << std::fixed << std::setprecision(1) << accuracy << "%" << std::endl;
    std::cout << "Final loss: " << std::fixed << std::setprecision(6)
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
    std::cout << "1. Non-linear Separability:" << std::endl;
    std::cout << "   XOR cannot be solved by a linear classifier (single-layer perceptron)." << std::endl;
    std::cout << "   A hidden layer enables the network to learn non-linear decision boundaries." << std::endl;
    std::cout << std::endl;

    std::cout << "2. Backpropagation:" << std::endl;
    std::cout << "   The network learns by computing gradients of the loss function" << std::endl;
    std::cout << "   with respect to all weights and biases, then updating them to minimize loss." << std::endl;
    std::cout << std::endl;

    std::cout << "3. Gradient Descent:" << std::endl;
    std::cout << "   Weights are updated using: w_new = w_old - learning_rate * gradient" << std::endl;
    std::cout << "   The learning rate controls how big each update step is." << std::endl;
    std::cout << std::endl;

    std::cout << "4. Activation Functions:" << std::endl;
    std::cout << "   Sigmoid activation introduces non-linearity, allowing the network" << std::endl;
    std::cout << "   to approximate complex functions like XOR." << std::endl;
    std::cout << std::endl;

    if (accuracy >= 90.0) {
        std::cout << "✓ SUCCESS: The network successfully learned the XOR function!" << std::endl;
        std::cout << "  With " << accuracy << "% accuracy, the network can correctly" << std::endl;
        std::cout << "  classify all (or nearly all) XOR input combinations." << std::endl;
    } else {
        std::cout << "⚠ PARTIAL SUCCESS: The network achieved " << accuracy << "% accuracy." << std::endl;
        std::cout << "  This may indicate:" << std::endl;
        std::cout << "  - More training epochs needed" << std::endl;
        std::cout << "  - Different learning rate required" << std::endl;
        std::cout << "  - Different weight initialization" << std::endl;
        std::cout << "  Try running again or adjusting hyperparameters." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
