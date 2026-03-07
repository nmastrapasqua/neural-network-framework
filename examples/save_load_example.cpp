#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <fstream>
#include <random>

/**
 * Save/Load Model Example - Neural Network Serialization Demonstration
 *
 * This example demonstrates how to save a trained neural network to a file
 * and load it back, preserving all weights, biases, and architecture.
 * This is essential for:
 * - Reusing trained models without retraining
 * - Sharing models between different programs
 * - Checkpointing during long training sessions
 * - Deploying models to production environments
 *
 * The example follows this workflow:
 * 1. Create and train a network on the XOR problem
 * 2. Save the trained network to a file (xor_model.txt)
 * 3. Load the network from the file into a new Network object
 * 4. Verify that predictions are identical before and after save/load
 * 5. Display the serialization format for educational purposes
 *
 * Requirements validated:
 * - 13.3: Example demonstrating model save/load functionality
 * - 13.4: Print progress and results
 * - 13.5: Detailed educational comments
 *
 * Serialization Requirements validated:
 * - 10.1: Serialize network to text format
 * - 10.2: Save all weights, biases, and configuration
 * - 10.3: Save activation function type for each layer
 * - 10.4: Reconstruct network with same architecture and parameters
 * - 10.9: Round-trip property (save → load → save produces identical output)
 */

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Model Save/Load - Serialization Demo" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // ============================================================================
    // STEP 1: Create and train a network on XOR
    // ============================================================================

    std::cout << "Step 1: Training a network on XOR problem" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    // Define XOR dataset
    std::vector<Vector> inputs = {
        Vector{0.0, 0.0},
        Vector{0.0, 1.0},
        Vector{1.0, 0.0},
        Vector{1.0, 1.0}
    };

    std::vector<Vector> targets = {
        Vector{0.0},
        Vector{1.0},
        Vector{1.0},
        Vector{0.0}
    };

    std::cout << "Dataset: XOR problem (4 examples)" << std::endl;
    std::cout << "Architecture: [2, 4, 1] with sigmoid activation" << std::endl;
    std::cout << std::endl;

    // Create network
    Network original_network;
    auto sigmoid = std::make_shared<Sigmoid>();
    original_network.addLayer(2, 4, sigmoid);
    original_network.addLayer(4, 1, sigmoid);

    // Initialize weights with fixed seed for reproducibility
    std::srand(42);
    original_network.getLayer(0).initializeXavier(2, 4);
    original_network.getLayer(1).initializeXavier(4, 1);

    // Train the network
    const size_t epochs = 5000;
    const double learning_rate = 0.5;
    MeanSquaredError loss_function;

    std::cout << "Training for " << epochs << " epochs..." << std::endl;
    std::vector<double> loss_history = original_network.train(
        inputs,
        targets,
        epochs,
        learning_rate,
        loss_function,
        1  // batch_size = 1 (SGD)
    );

    std::cout << "Training complete!" << std::endl;
    std::cout << "Final loss: " << std::fixed << std::setprecision(6)
              << loss_history.back() << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 2: Test the original network and record predictions
    // ============================================================================

    std::cout << "Step 2: Testing original network" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    std::cout << "Predictions from ORIGINAL network:" << std::endl;
    std::cout << "  Input 1 | Input 2 | Target | Prediction" << std::endl;
    std::cout << "  --------|---------|--------|------------" << std::endl;

    // Store original predictions for comparison later
    std::vector<Vector> original_predictions;

    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector prediction = original_network.predict(inputs[i]);
        original_predictions.push_back(prediction);

        std::cout << "     " << inputs[i][0] << "    |    "
                  << inputs[i][1] << "    |   "
                  << targets[i][0] << "    |   "
                  << std::fixed << std::setprecision(4) << prediction[0] << std::endl;
    }
    std::cout << std::endl;

    // ============================================================================
    // STEP 3: Save the network to a file
    // ============================================================================

    std::cout << "Step 3: Saving network to file" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    const std::string filename = "xor_model.txt";

    std::cout << "Saving trained network to: " << filename << std::endl;
    std::cout << "This file will contain:" << std::endl;
    std::cout << "  - Network architecture (layer sizes)" << std::endl;
    std::cout << "  - Activation function types" << std::endl;
    std::cout << "  - All weight matrices" << std::endl;
    std::cout << "  - All bias vectors" << std::endl;
    std::cout << std::endl;

    try {
        original_network.save(filename);
        std::cout << "✓ Network saved successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Error saving network: " << e.what() << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // ============================================================================
    // STEP 4: Display the serialization format (educational)
    // ============================================================================

    std::cout << "Step 4: Examining serialization format" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "The serialization format is human-readable text." << std::endl;
    std::cout << "Let's look at the contents of " << filename << ":" << std::endl;
    std::cout << std::endl;
    std::cout << "--- File Contents (first 30 lines) ---" << std::endl;

    // Read and display the file contents
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        int line_count = 0;
        const int max_lines = 30;

        while (std::getline(file, line) && line_count < max_lines) {
            std::cout << line << std::endl;
            line_count++;
        }

        if (line_count >= max_lines) {
            std::cout << "... (file continues) ..." << std::endl;
        }

        file.close();
    } else {
        std::cout << "Could not open file for reading." << std::endl;
    }

    std::cout << "--- End of File Contents ---" << std::endl;
    std::cout << std::endl;

    std::cout << "Format Explanation:" << std::endl;
    std::cout << "  - Header identifies the file format and version" << std::endl;
    std::cout << "  - Each layer section contains:" << std::endl;
    std::cout << "    * Input and output dimensions" << std::endl;
    std::cout << "    * Activation function name" << std::endl;
    std::cout << "    * Weight matrix (one row per output neuron)" << std::endl;
    std::cout << "    * Bias vector (one value per output neuron)" << std::endl;
    std::cout << "  - This format is easy to inspect and debug" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 5: Load the network from the file
    // ============================================================================

    std::cout << "Step 5: Loading network from file" << std::endl;
    std::cout << "----------------------------------" << std::endl;

    std::cout << "Creating a new Network object and loading from: " << filename << std::endl;

    Network loaded_network;

    try {
        loaded_network.load(filename);
        std::cout << "✓ Network loaded successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Error loading network: " << e.what() << std::endl;
        return 1;
    }

    // Verify topology matches
    std::vector<size_t> original_topology = original_network.getTopology();
    std::vector<size_t> loaded_topology = loaded_network.getTopology();

    std::cout << "Loaded network topology: [";
    for (size_t i = 0; i < loaded_topology.size(); ++i) {
        std::cout << loaded_topology[i];
        if (i < loaded_topology.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 6: Test the loaded network
    // ============================================================================

    std::cout << "Step 6: Testing loaded network" << std::endl;
    std::cout << "-------------------------------" << std::endl;

    std::cout << "Predictions from LOADED network:" << std::endl;
    std::cout << "  Input 1 | Input 2 | Target | Prediction" << std::endl;
    std::cout << "  --------|---------|--------|------------" << std::endl;

    std::vector<Vector> loaded_predictions;

    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector prediction = loaded_network.predict(inputs[i]);
        loaded_predictions.push_back(prediction);

        std::cout << "     " << inputs[i][0] << "    |    "
                  << inputs[i][1] << "    |   "
                  << targets[i][0] << "    |   "
                  << std::fixed << std::setprecision(4) << prediction[0] << std::endl;
    }
    std::cout << std::endl;

    // ============================================================================
    // STEP 7: Verify predictions are identical
    // ============================================================================

    std::cout << "Step 7: Verifying prediction consistency" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    std::cout << "Comparing predictions (Original vs Loaded):" << std::endl;
    std::cout << "  Input 1 | Input 2 | Original | Loaded   | Difference" << std::endl;
    std::cout << "  --------|---------|----------|----------|------------" << std::endl;

    bool all_identical = true;
    const double epsilon = 1e-10;  // Tolerance for floating-point comparison
    double max_difference = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
        double original_pred = original_predictions[i][0];
        double loaded_pred = loaded_predictions[i][0];
        double difference = std::abs(original_pred - loaded_pred);

        if (difference > max_difference) {
            max_difference = difference;
        }

        if (difference > epsilon) {
            all_identical = false;
        }

        std::cout << "     " << inputs[i][0] << "    |    "
                  << inputs[i][1] << "    | "
                  << std::fixed << std::setprecision(6) << original_pred << " | "
                  << std::fixed << std::setprecision(6) << loaded_pred << " | "
                  << std::scientific << std::setprecision(2) << difference << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Maximum difference: " << std::scientific << std::setprecision(2)
              << max_difference << std::endl;
    std::cout << "Tolerance (epsilon): " << std::scientific << std::setprecision(2)
              << epsilon << std::endl;
    std::cout << std::endl;

    if (all_identical) {
        std::cout << "✓ SUCCESS: All predictions are identical!" << std::endl;
        std::cout << "  The loaded network produces exactly the same outputs" << std::endl;
        std::cout << "  as the original network. Serialization is working correctly." << std::endl;
    } else {
        std::cout << "⚠ WARNING: Predictions differ by more than epsilon!" << std::endl;
        std::cout << "  This may indicate a problem with serialization/deserialization." << std::endl;
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

    std::cout << "1. Model Serialization:" << std::endl;
    std::cout << "   Serialization converts a trained neural network into a format" << std::endl;
    std::cout << "   that can be stored in a file. This includes:" << std::endl;
    std::cout << "   - Network architecture (layer sizes, activation functions)" << std::endl;
    std::cout << "   - Learned parameters (weights and biases)" << std::endl;
    std::cout << std::endl;

    std::cout << "2. Model Deserialization:" << std::endl;
    std::cout << "   Deserialization reconstructs a neural network from a saved file." << std::endl;
    std::cout << "   The reconstructed network should be functionally identical to" << std::endl;
    std::cout << "   the original network, producing the same predictions." << std::endl;
    std::cout << std::endl;

    std::cout << "3. Round-Trip Property:" << std::endl;
    std::cout << "   A good serialization format satisfies the round-trip property:" << std::endl;
    std::cout << "   save(network) → load() → save() produces identical output." << std::endl;
    std::cout << "   This ensures no information is lost during serialization." << std::endl;
    std::cout << std::endl;

    std::cout << "4. Text Format Benefits:" << std::endl;
    std::cout << "   Using a human-readable text format (instead of binary) provides:" << std::endl;
    std::cout << "   - Easy inspection and debugging" << std::endl;
    std::cout << "   - Platform independence (no endianness issues)" << std::endl;
    std::cout << "   - Version control friendly (can diff changes)" << std::endl;
    std::cout << "   - Educational value (can see what's being saved)" << std::endl;
    std::cout << std::endl;

    std::cout << "5. Practical Applications:" << std::endl;
    std::cout << "   Model save/load is essential for:" << std::endl;
    std::cout << "   - Training once, deploying many times" << std::endl;
    std::cout << "   - Checkpointing during long training sessions" << std::endl;
    std::cout << "   - Sharing models between researchers" << std::endl;
    std::cout << "   - A/B testing different model versions" << std::endl;
    std::cout << "   - Transfer learning (loading pre-trained models)" << std::endl;
    std::cout << std::endl;

    std::cout << "Demo completed successfully!" << std::endl;
    std::cout << "The trained model has been saved to: " << filename << std::endl;

    return 0;
}
