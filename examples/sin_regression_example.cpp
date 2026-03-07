#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <cmath>
#include <random>

/**
 * Sin(x) Function Approximation Example - Regression Demonstration
 *
 * This example demonstrates **regression** (predicting continuous values)
 * instead of classification. The network learns to approximate the sin(x) function.
 *
 * Problem Setup:
 * - Input: x value in range [-π, π]
 * - Output: sin(x) value in range [-1, 1]
 * - Task: Learn the mapping x → sin(x)
 *
 * Network Architecture:
 * - Input layer: 1 neuron (x value)
 * - Hidden layer 1: 16 neurons (tanh activation)
 * - Hidden layer 2: 16 neurons (tanh activation)
 * - Output layer: 1 neuron (tanh activation, outputs in [-1, 1])
 *
 * Key Concepts Demonstrated:
 * - **Regression** vs Classification (continuous output vs discrete classes)
 * - **Universal Function Approximation**: Neural networks can approximate any continuous function
 * - **Interpolation** vs **Extrapolation**: How well does the network generalize?
 * - **Overfitting** on functions: Too many neurons → memorizes instead of learning pattern
 *
 * Educational Value:
 * - Shows neural networks can learn mathematical functions
 * - Demonstrates the power and limitations of function approximation
 * - Easy to visualize (can plot learned function vs actual)
 * - Highlights importance of training range
 */

const double PI = 3.14159265358979323846;

/**
 * Generate training data for sin(x)
 *
 * @param num_samples Number of training samples
 * @param x_min Minimum x value
 * @param x_max Maximum x value
 * @param add_noise Whether to add small noise to outputs (makes learning more robust)
 */
void generateSinData(std::vector<Vector>& inputs,
                    std::vector<Vector>& targets,
                    size_t num_samples,
                    double x_min,
                    double x_max,
                    bool add_noise = false,
                    unsigned int seed = 42) {
    inputs.clear();
    targets.clear();

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> noise_dist(-0.05, 0.05);

    // Generate evenly spaced samples
    for (size_t i = 0; i < num_samples; ++i) {
        double x = x_min + (x_max - x_min) * i / (num_samples - 1);
        double y = std::sin(x);

        // Add small noise if requested
        if (add_noise) {
            y += noise_dist(rng);
            // Clamp to [-1, 1] since sin output is in this range
            y = std::max(-1.0, std::min(1.0, y));
        }

        inputs.push_back(Vector{x});
        targets.push_back(Vector{y});
    }
}

/**
 * Calculate Mean Absolute Error (MAE)
 * MAE is often more interpretable than MSE for regression
 */
double calculateMAE(Network& network,
                   const std::vector<Vector>& inputs,
                   const std::vector<Vector>& targets) {
    double total_error = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector prediction = network.predict(inputs[i]);
        double error = std::abs(prediction[0] - targets[i][0]);
        total_error += error;
    }

    return total_error / inputs.size();
}

/**
 * Display a simple ASCII plot of the function
 */
void plotFunction(Network& network,
                 double x_min,
                 double x_max,
                 size_t num_points = 80) {
    const int plot_height = 25;
    const int plot_width = 80;

    // Generate points
    std::vector<double> x_values;
    std::vector<double> actual_values;
    std::vector<double> predicted_values;

    for (size_t i = 0; i < num_points; ++i) {
        double x = x_min + (x_max - x_min) * i / (num_points - 1);
        double actual = std::sin(x);
        Vector prediction = network.predict(Vector{x});

        x_values.push_back(x);
        actual_values.push_back(actual);
        predicted_values.push_back(prediction[0]);
    }

    // Create plot grid
    std::vector<std::vector<char>> grid(plot_height, std::vector<char>(plot_width, ' '));

    // Draw axes
    int mid_row = plot_height / 2;
    for (int col = 0; col < plot_width; ++col) {
        grid[mid_row][col] = '-';
    }

    // Plot actual function (with 'o')
    for (size_t i = 0; i < num_points; ++i) {
        int col = static_cast<int>((x_values[i] - x_min) / (x_max - x_min) * (plot_width - 1));
        int row = mid_row - static_cast<int>(actual_values[i] * (plot_height / 2 - 1));

        if (row >= 0 && row < plot_height && col >= 0 && col < plot_width) {
            grid[row][col] = 'o';
        }
    }

    // Plot predicted function (with '*')
    for (size_t i = 0; i < num_points; ++i) {
        int col = static_cast<int>((x_values[i] - x_min) / (x_max - x_min) * (plot_width - 1));
        int row = mid_row - static_cast<int>(predicted_values[i] * (plot_height / 2 - 1));

        if (row >= 0 && row < plot_height && col >= 0 && col < plot_width) {
            if (grid[row][col] == 'o') {
                grid[row][col] = '@';  // Overlap
            } else {
                grid[row][col] = '*';
            }
        }
    }

    // Print grid with border
    std::cout << "       +" << std::string(plot_width, '-') << "+" << std::endl;

    for (int row = 0; row < plot_height; ++row) {
        // Y-axis labels
        if (row == 0) {
            std::cout << "  1.0  |";
        } else if (row == mid_row) {
            std::cout << "  0.0  |";
        } else if (row == plot_height - 1) {
            std::cout << " -1.0  |";
        } else {
            std::cout << "       |";
        }

        // Plot content
        for (int col = 0; col < plot_width; ++col) {
            std::cout << grid[row][col];
        }
        std::cout << "|" << std::endl;
    }

    std::cout << "       +" << std::string(plot_width, '-') << "+" << std::endl;

    // X-axis labels
    std::cout << "       ";
    std::cout << std::fixed << std::setprecision(2) << x_min;
    int spacing = plot_width - 10;
    for (int i = 0; i < spacing; ++i) std::cout << " ";
    std::cout << std::fixed << std::setprecision(2) << x_max << std::endl;

    std::cout << std::endl;
    std::cout << "       Legend: o = actual sin(x), * = predicted, @ = overlap (perfect match)" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Sin(x) Function Approximation Demo" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // ============================================================================
    // STEP 1: Generate training data
    // ============================================================================

    std::cout << "Step 1: Generating training data" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::vector<Vector> train_inputs;
    std::vector<Vector> train_targets;

    // Training range: [-π, π]
    const double train_x_min = -PI;
    const double train_x_max = PI;
    const size_t num_train_samples = 100;

    generateSinData(train_inputs, train_targets,
                   num_train_samples, train_x_min, train_x_max,
                   false, 42);  // No noise for cleaner learning

    std::cout << "Generated " << num_train_samples << " training samples" << std::endl;
    std::cout << "Training range: [" << std::fixed << std::setprecision(2)
              << train_x_min << ", " << train_x_max << "]" << std::endl;
    std::cout << "Function: y = sin(x)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 2: Generate test data (including extrapolation range)
    // ============================================================================

    std::cout << "Step 2: Generating test data" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    std::vector<Vector> test_inputs;
    std::vector<Vector> test_targets;

    // Test range: [-2π, 2π] (wider than training to test extrapolation)
    const double test_x_min = -2 * PI;
    const double test_x_max = 2 * PI;
    const size_t num_test_samples = 200;

    generateSinData(test_inputs, test_targets,
                   num_test_samples, test_x_min, test_x_max,
                   false, 123);

    std::cout << "Generated " << num_test_samples << " test samples" << std::endl;
    std::cout << "Test range: [" << std::fixed << std::setprecision(2)
              << test_x_min << ", " << test_x_max << "]" << std::endl;
    std::cout << "Note: Test range is WIDER than training range" << std::endl;
    std::cout << "      This tests extrapolation ability!" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 3: Create neural network
    // ============================================================================

    std::cout << "Step 3: Creating neural network" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    Network network;
    auto tanh_activation = std::make_shared<Tanh>();

    // Architecture: [1, 16, 16, 1]
    // - 1 input (x value)
    // - 16 hidden neurons (layer 1)
    // - 16 hidden neurons (layer 2)
    // - 1 output (sin(x) value)
    // Using tanh because output range is [-1, 1], same as sin(x)
    network.addLayer(1, 16, tanh_activation);
    network.addLayer(16, 16, tanh_activation);
    network.addLayer(16, 1, tanh_activation);

    std::cout << "Network architecture: [1, 16, 16, 1]" << std::endl;
    std::cout << "  - Input layer: 1 neuron (x value)" << std::endl;
    std::cout << "  - Hidden layer 1: 16 neurons (tanh)" << std::endl;
    std::cout << "  - Hidden layer 2: 16 neurons (tanh)" << std::endl;
    std::cout << "  - Output layer: 1 neuron (tanh, outputs in [-1, 1])" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 4: Initialize weights
    // ============================================================================

    std::cout << "Step 4: Initializing weights" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    std::srand(42);
    network.getLayer(0).initializeXavier(1, 16);
    network.getLayer(1).initializeXavier(16, 16);
    network.getLayer(2).initializeXavier(16, 1);

    std::cout << "Weights initialized with Xavier initialization" << std::endl;
    std::cout << "Random seed: 42 (for reproducibility)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 5: Train the network
    // ============================================================================

    std::cout << "Step 5: Training the network" << std::endl;
    std::cout << "----------------------------" << std::endl;

    const size_t epochs = 1000;
    const double learning_rate = 0.01;
    MeanSquaredError loss_function;

    std::cout << "Training configuration:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Learning rate: " << learning_rate << std::endl;
    std::cout << "  - Loss function: MSE" << std::endl;
    std::cout << "  - Batch size: 1 (SGD)" << std::endl;
    std::cout << std::endl;

    std::cout << "Training progress (every 200 epochs):" << std::endl;
    std::cout << "Epoch | Train Loss | Train MAE | Test MAE" << std::endl;
    std::cout << "------|------------|-----------|----------" << std::endl;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Train for one epoch
        std::vector<double> loss_history = network.train(
            train_inputs,
            train_targets,
            1,
            learning_rate,
            loss_function,
            1
        );

        // Display progress every 200 epochs
        if ((epoch + 1) % 200 == 0 || epoch == 0) {
            double train_mae = calculateMAE(network, train_inputs, train_targets);
            double test_mae = calculateMAE(network, test_inputs, test_targets);

            std::cout << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(6) << loss_history[0] << " | "
                      << std::fixed << std::setprecision(6) << train_mae << " | "
                      << std::fixed << std::setprecision(6) << test_mae << std::endl;
        }
    }

    std::cout << std::endl;

    // ============================================================================
    // STEP 6: Evaluate on training range (interpolation)
    // ============================================================================

    std::cout << "Step 6: Evaluation on training range (interpolation)" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    // Create test set within training range
    std::vector<Vector> interp_inputs;
    std::vector<Vector> interp_targets;
    generateSinData(interp_inputs, interp_targets, 50, train_x_min, train_x_max, false, 999);

    double interp_mae = calculateMAE(network, interp_inputs, interp_targets);

    std::cout << "Interpolation MAE: " << std::fixed << std::setprecision(6) << interp_mae << std::endl;
    std::cout << "This measures accuracy within the training range." << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 7: Evaluate on extended range (extrapolation)
    // ============================================================================

    std::cout << "Step 7: Evaluation on extended range (extrapolation)" << std::endl;
    std::cout << "-----------------------------------------------------" << std::endl;

    // Create test set outside training range
    std::vector<Vector> extrap_inputs;
    std::vector<Vector> extrap_targets;

    // Left extrapolation: [-2π, -π]
    generateSinData(extrap_inputs, extrap_targets, 25, -2*PI, -PI, false, 888);

    // Right extrapolation: [π, 2π]
    std::vector<Vector> right_inputs, right_targets;
    generateSinData(right_inputs, right_targets, 25, PI, 2*PI, false, 777);
    extrap_inputs.insert(extrap_inputs.end(), right_inputs.begin(), right_inputs.end());
    extrap_targets.insert(extrap_targets.end(), right_targets.begin(), right_targets.end());

    double extrap_mae = calculateMAE(network, extrap_inputs, extrap_targets);

    std::cout << "Extrapolation MAE: " << std::fixed << std::setprecision(6) << extrap_mae << std::endl;
    std::cout << "This measures accuracy OUTSIDE the training range." << std::endl;
    std::cout << "Neural networks typically struggle with extrapolation!" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 8: Visualize learned function
    // ============================================================================

    std::cout << "Step 8: Visualizing learned function" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    std::cout << std::endl;

    std::cout << "ASCII Plot - Training Range [-π, π]:" << std::endl;
    std::cout << "(This shows how well the network learned within the training data)" << std::endl;
    std::cout << std::endl;
    plotFunction(network, train_x_min, train_x_max, 80);
    std::cout << std::endl;

    std::cout << "ASCII Plot - Extended Range [-2π, 2π]:" << std::endl;
    std::cout << "(This includes extrapolation beyond the training range)" << std::endl;
    std::cout << std::endl;
    plotFunction(network, test_x_min, test_x_max, 80);
    std::cout << std::endl;

    // ============================================================================
    // STEP 9: Show sample predictions
    // ============================================================================

    std::cout << "Step 9: Sample predictions" << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << std::endl;

    std::cout << "   x    | Actual sin(x) | Predicted | Error" << std::endl;
    std::cout << "--------|---------------|-----------|-------" << std::endl;

    std::vector<double> test_x_values = {-2*PI, -PI, -PI/2, 0, PI/2, PI, 2*PI};
    for (double x : test_x_values) {
        double actual = std::sin(x);
        Vector prediction = network.predict(Vector{x});
        double error = std::abs(prediction[0] - actual);

        std::cout << std::fixed << std::setprecision(3) << std::setw(7) << x << " | "
                  << std::setw(13) << actual << " | "
                  << std::setw(9) << prediction[0] << " | "
                  << std::setw(5) << error << std::endl;
    }
    std::cout << std::endl;

    // ============================================================================
    // Educational Summary
    // ============================================================================

    std::cout << "========================================" << std::endl;
    std::cout << "  Educational Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Key Concepts Demonstrated:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Regression vs Classification:" << std::endl;
    std::cout << "   - Regression: Predict continuous values (sin(x) in [-1, 1])" << std::endl;
    std::cout << "   - Classification: Predict discrete classes (like MNIST digits)" << std::endl;
    std::cout << "   - Same network architecture, different interpretation!" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Universal Function Approximation:" << std::endl;
    std::cout << "   Neural networks can approximate ANY continuous function!" << std::endl;
    std::cout << "   This is a fundamental theorem in neural network theory." << std::endl;
    std::cout << "   We demonstrated this by learning sin(x)." << std::endl;
    std::cout << std::endl;

    std::cout << "3. Interpolation vs Extrapolation:" << std::endl;
    std::cout << "   - Interpolation MAE: " << std::fixed << std::setprecision(4) << interp_mae << std::endl;
    std::cout << "   - Extrapolation MAE: " << std::fixed << std::setprecision(4) << extrap_mae << std::endl;

    if (extrap_mae > interp_mae * 2) {
        std::cout << "   Neural networks struggle with extrapolation!" << std::endl;
        std::cout << "   They work best within the training data range." << std::endl;
    } else {
        std::cout << "   The network generalized reasonably well!" << std::endl;
        std::cout << "   Sin(x) is periodic, which helps extrapolation." << std::endl;
    }
    std::cout << std::endl;

    std::cout << "4. Activation Function Choice:" << std::endl;
    std::cout << "   We used tanh for all layers because:" << std::endl;
    std::cout << "   - Output range [-1, 1] matches sin(x) range" << std::endl;
    std::cout << "   - Zero-centered (better for learning)" << std::endl;
    std::cout << std::endl;

    std::cout << "5. Practical Applications:" << std::endl;
    std::cout << "   Function approximation is useful for:" << std::endl;
    std::cout << "   - Time series prediction (stock prices, weather)" << std::endl;
    std::cout << "   - Physics simulations (approximate complex equations)" << std::endl;
    std::cout << "   - Control systems (learn system dynamics)" << std::endl;
    std::cout << std::endl;

    if (interp_mae < 0.05) {
        std::cout << "✓ EXCELLENT: Interpolation MAE < 0.05!" << std::endl;
        std::cout << "  The network learned sin(x) very accurately." << std::endl;
    } else if (interp_mae < 0.1) {
        std::cout << "✓ GOOD: Interpolation MAE < 0.1" << std::endl;
        std::cout << "  The network learned a reasonable approximation." << std::endl;
    } else {
        std::cout << "⚠ NEEDS IMPROVEMENT: Interpolation MAE > 0.1" << std::endl;
        std::cout << "  Try more epochs, more neurons, or lower learning rate." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
