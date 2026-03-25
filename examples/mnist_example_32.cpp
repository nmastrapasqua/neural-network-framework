#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include "mnist_loader.h"
#include <iostream>
#include <memory>
#include <iomanip>
#include <random>
#include <chrono>

/**
 * MNIST Handwritten Digit Recognition Example
 *
 * This example demonstrates training a neural network on the MNIST dataset,
 * which is a classic benchmark in machine learning. MNIST contains 70,000
 * grayscale images of handwritten digits (0-9), each 28x28 pixels.
 *
 * Dataset Structure:
 * - Training set: 60,000 images (we'll use a subset of 10,000 for faster training)
 * - Test set: 10,000 images
 * - Image size: 28x28 pixels = 784 input features
 * - Classes: 10 (digits 0-9)
 *
 * Network Architecture:
 * - Input layer: 784 neurons (one per pixel)
 * - Hidden layer 1: 128 neurons (ReLU activation)
 * - Hidden layer 2: 64 neurons (ReLU activation)
 * - Output layer: 10 neurons (sigmoid activation, one per digit class)
 *
 * Training Configuration:
 * - Epochs: 10
 * - Learning rate: 0.01
 * - Loss function: Mean Squared Error (MSE)
 * - Subset size: 10,000 training images (for faster training)
 *
 * Expected Performance:
 * - Target accuracy: >85% on test set
 * - Training time: A few minutes on modern hardware
 *
 * Requirements validated:
 * - 13.4: Display training progress (loss and accuracy per epoch)
 * - 13.5: Detailed educational comments explaining the implementation
 */

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  MNIST Digit Recognition Demo" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // ============================================================================
    // STEP 1: Load MNIST dataset
    // ============================================================================

    std::cout << "Step 1: Loading MNIST dataset" << std::endl;
    std::cout << "------------------------------" << std::endl;

    std::vector<Vector> train_images;
    std::vector<Vector> train_labels;
    std::vector<Vector> test_images;
    std::vector<Vector> test_labels;

    try {
        // Load training data
        std::cout << "Loading training images..." << std::endl;
        MNISTLoader::loadImages("mnist-dataset/train-images.idx3-ubyte", train_images);
        std::cout << "Loading training labels..." << std::endl;
        MNISTLoader::loadLabels("mnist-dataset/train-labels.idx1-ubyte", train_labels);

        // Load test data
        std::cout << "Loading test images..." << std::endl;
        MNISTLoader::loadImages("mnist-dataset/t10k-images.idx3-ubyte", test_images);
        std::cout << "Loading test labels..." << std::endl;
        MNISTLoader::loadLabels("mnist-dataset/t10k-labels.idx1-ubyte", test_labels);

        std::cout << std::endl;
        std::cout << "Dataset loaded successfully!" << std::endl;
        std::cout << "  - Training images: " << train_images.size() << std::endl;
        std::cout << "  - Training labels: " << train_labels.size() << std::endl;
        std::cout << "  - Test images: " << test_images.size() << std::endl;
        std::cout << "  - Test labels: " << test_labels.size() << std::endl;
        std::cout << "  - Image dimensions: 28x28 = 784 pixels" << std::endl;
        std::cout << "  - Number of classes: 10 (digits 0-9)" << std::endl;
        std::cout << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error loading MNIST dataset: " << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << "Please ensure the MNIST dataset files are in the 'mnist-dataset/' directory:" << std::endl;
        std::cerr << "  - mnist-dataset/train-images.idx3-ubyte" << std::endl;
        std::cerr << "  - mnist-dataset/train-labels.idx1-ubyte" << std::endl;
        std::cerr << "  - mnist-dataset/t10k-images.idx3-ubyte" << std::endl;
        std::cerr << "  - mnist-dataset/t10k-labels.idx1-ubyte" << std::endl;
        return 1;
    }

    // ============================================================================
    // STEP 2: Create training subset for faster training
    // ============================================================================

    std::cout << "Step 2: Creating training subset" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    // Use only 10,000 images instead of 60,000 for faster training
    // This is a common practice for quick experiments and demonstrations
    const size_t subset_size = 10000;

    if (train_images.size() > subset_size) {
        train_images.erase(train_images.begin() + subset_size, train_images.end());
        train_labels.erase(train_labels.begin() + subset_size, train_labels.end());
        std::cout << "Using subset of " << subset_size << " training images for faster training" << std::endl;
    } else {
        std::cout << "Using all " << train_images.size() << " training images" << std::endl;
    }
    std::cout << std::endl;

    // ============================================================================
    // STEP 3: Display sample images (optional, educational)
    // ============================================================================

    std::cout << "Step 3: Displaying sample images" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    std::cout << "Here are a few examples from the training set:" << std::endl;
    std::cout << std::endl;

    // Display first 3 training images as ASCII art
    const size_t num_samples = 3;
    for (size_t i = 0; i < num_samples && i < train_images.size(); ++i) {
        std::cout << "Training example " << (i + 1) << ":" << std::endl;
        MNISTLoader::displayImage(train_images[i], &train_labels[i]);
        std::cout << std::endl;
    }

    // ============================================================================
    // STEP 4: Create the neural network architecture
    // ============================================================================

    std::cout << "Step 4: Creating network architecture" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // Create a network with architecture: [784, 128, 64, 10]
    // - 784 input neurons (28x28 pixels)
    // - 128 hidden neurons (first hidden layer)
    // - 64 hidden neurons (second hidden layer)
    // - 10 output neurons (one per digit class 0-9)
    Network network;

    // Use ReLU activation for hidden layers
    // ReLU: f(x) = max(0, x)
    // ReLU is the most popular activation for deep networks because:
    // - It helps avoid vanishing gradient problem
    // - It's computationally efficient
    // - It often leads to faster convergence
    auto relu = std::make_shared<ReLU>();

    // Use sigmoid activation for output layer
    // Sigmoid: σ(x) = 1 / (1 + e^(-x))
    // Output range: (0, 1), suitable for multi-class classification with MSE loss
    auto sigmoid = std::make_shared<Sigmoid>();

    // Add layers to the network
    network.addLayer(784, 128, relu);      // Input -> Hidden 1
    network.addLayer(128, 64, relu);       // Hidden 1 -> Hidden 2
    network.addLayer(64, 10, sigmoid);     // Hidden 2 -> Output

    std::cout << "Network architecture: [784, 128, 64, 10]" << std::endl;
    std::cout << "  - Input layer: 784 neurons (28x28 pixels)" << std::endl;
    std::cout << "  - Hidden layer 1: 128 neurons (ReLU activation)" << std::endl;
    std::cout << "  - Hidden layer 2: 64 neurons (ReLU activation)" << std::endl;
    std::cout << "  - Output layer: 10 neurons (sigmoid activation)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 5: Initialize weights
    // ============================================================================

    std::cout << "Step 5: Initializing weights" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    // Set a fixed random seed for reproducibility
    Matrix::setSeed(42);

    // Initialize weights using He initialization for ReLU layers
    // He initialization: weights ~ Uniform(-√(2/fan_in), √(2/fan_in))
    // This initialization is specifically designed for ReLU activation
    network.getLayer(0).initializeHe(784);   // Hidden layer 1
    network.getLayer(1).initializeHe(128);   // Hidden layer 2

    // Initialize weights using Xavier initialization for sigmoid output layer
    // Xavier initialization: weights ~ Uniform(-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out)))
    network.getLayer(2).initializeXavier(64, 10);  // Output layer

    std::cout << "Weights initialized:" << std::endl;
    std::cout << "  - Hidden layers: He initialization (optimized for ReLU)" << std::endl;
    std::cout << "  - Output layer: Xavier initialization (optimized for sigmoid)" << std::endl;
    std::cout << "  - Random seed: 42 (for reproducibility)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 6: Set up training parameters
    // ============================================================================

    std::cout << "Step 6: Configuring training parameters" << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    const size_t epochs = 20;              // Number of complete passes through the dataset
    const double learning_rate = 0.1;     // Step size for gradient descent

    std::cout << "Training configuration:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Learning rate: " << learning_rate << std::endl;
    std::cout << "  - Loss function: Mean Squared Error (MSE)" << std::endl;
    std::cout << "  - Optimization: mini batch" << std::endl;
    std::cout << "  - Training samples: " << train_images.size() << std::endl;
    std::cout << "  - Test samples: " << test_images.size() << std::endl;
    std::cout << std::endl;

    // Create loss function
    MeanSquaredError loss_function;

    // ============================================================================
    // STEP 7: Train the network
    // ============================================================================

    std::cout << "Step 7: Training the network" << std::endl;
    std::cout << "----------------------------" << std::endl;
    std::cout << "Training in progress (this may take a few minutes)..." << std::endl;
    std::cout << std::endl;

    // Record start time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Train the network
    // We'll manually implement the training loop to display progress after each epoch
    std::cout << "Epoch | Train Loss | Train Acc | Test Acc  | Time (s)" << std::endl;
    std::cout << "------|------------|-----------|-----------|----------" << std::endl;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Train for one epoch
        std::vector<double> loss_history = network.train(
            train_images,
            train_labels,
            1,  // Train for 1 epoch at a time
            learning_rate,
            loss_function,
            32   // batch_size = 32
        );

        // Calculate training accuracy
        double train_accuracy = network.calculateAccuracy(train_images, train_labels);

        // Calculate test accuracy
        double test_accuracy = network.calculateAccuracy(test_images, test_labels);

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - epoch_start);

        // Display progress
        std::cout << std::setw(5) << (epoch + 1) << " | "
                  << std::fixed << std::setprecision(6) << loss_history[0] << " | "
                  << std::fixed << std::setprecision(2) << (train_accuracy * 100) << "%   | "
                  << std::fixed << std::setprecision(2) << (test_accuracy * 100) << "%    | "
                  << epoch_duration.count() << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);

    std::cout << std::endl;
    std::cout << "Training completed in " << total_duration.count() << " seconds" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 8: Final evaluation on test set
    // ============================================================================

    std::cout << "Step 8: Final evaluation on test set" << std::endl;
    std::cout << "--------------------------------------" << std::endl;

    // Calculate final test accuracy
    double final_test_accuracy = network.calculateAccuracy(test_images, test_labels);
    double final_test_loss = network.validate(test_images, test_labels, loss_function);

    std::cout << "Final test set performance:" << std::endl;
    std::cout << "  - Test accuracy: " << std::fixed << std::setprecision(2)
              << (final_test_accuracy * 100) << "%" << std::endl;
    std::cout << "  - Test loss: " << std::fixed << std::setprecision(6)
              << final_test_loss << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 9: Display some predictions with visualizations
    // ============================================================================

    std::cout << "Step 9: Sample predictions with visualizations" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "Here are some predictions on test images:" << std::endl;
    std::cout << std::endl;

    // Display predictions for first 5 test images
    const size_t num_predictions = 5;
    for (size_t i = 0; i < num_predictions && i < test_images.size(); ++i) {
        // Make prediction
        Vector prediction = network.predict(test_images[i]);

        // Get predicted class (index of maximum output)
        uint8_t predicted_class = MNISTLoader::oneHotToLabel(prediction);
        uint8_t actual_class = MNISTLoader::oneHotToLabel(test_labels[i]);

        // Display image and prediction
        std::cout << "Test example " << (i + 1) << ":" << std::endl;
        std::cout << "Actual label: " << static_cast<int>(actual_class) << std::endl;
        std::cout << "Predicted label: " << static_cast<int>(predicted_class);

        if (predicted_class == actual_class) {
            std::cout << " ✓ CORRECT" << std::endl;
        } else {
            std::cout << " ✗ INCORRECT" << std::endl;
        }

        std::cout << std::endl;
        std::cout << "Output probabilities:" << std::endl;
        std::cout << "  ";
        for (size_t j = 0; j < 10; ++j) {
            std::cout << j << ": " << std::fixed << std::setprecision(3) << prediction[j];
            if (j < 9) std::cout << ", ";
        }
        std::cout << std::endl;
        std::cout << std::endl;

        // Display the image
        MNISTLoader::displayImage(test_images[i]);
        std::cout << std::endl;
    }

    // ============================================================================
    // Educational Summary
    // ============================================================================

    std::cout << "========================================" << std::endl;
    std::cout << "  Educational Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << "Key Concepts Demonstrated:" << std::endl;
    std::cout << std::endl;

    std::cout << "1. Multi-Class Classification:" << std::endl;
    std::cout << "   MNIST is a 10-class classification problem (digits 0-9)." << std::endl;
    std::cout << "   The network outputs 10 values, one for each class." << std::endl;
    std::cout << "   The predicted class is the one with the highest output value." << std::endl;
    std::cout << std::endl;

    std::cout << "2. Deep Neural Network:" << std::endl;
    std::cout << "   This network has 2 hidden layers, making it a 'deep' network." << std::endl;
    std::cout << "   Multiple hidden layers allow the network to learn hierarchical features:" << std::endl;
    std::cout << "   - First layer: learns simple edges and curves" << std::endl;
    std::cout << "   - Second layer: combines edges into more complex shapes" << std::endl;
    std::cout << "   - Output layer: combines shapes to recognize digits" << std::endl;
    std::cout << std::endl;

    std::cout << "3. ReLU Activation:" << std::endl;
    std::cout << "   ReLU (Rectified Linear Unit) is used in hidden layers because:" << std::endl;
    std::cout << "   - It helps prevent vanishing gradient problem in deep networks" << std::endl;
    std::cout << "   - It's computationally efficient (just max(0, x))" << std::endl;
    std::cout << "   - It often leads to faster training convergence" << std::endl;
    std::cout << std::endl;

    std::cout << "4. Weight Initialization:" << std::endl;
    std::cout << "   Different activation functions require different initialization strategies:" << std::endl;
    std::cout << "   - He initialization for ReLU layers (accounts for ReLU's properties)" << std::endl;
    std::cout << "   - Xavier initialization for sigmoid layers (balanced variance)" << std::endl;
    std::cout << std::endl;

    std::cout << "5. Training on Real Data:" << std::endl;
    std::cout << "   Unlike XOR (4 examples), MNIST has thousands of examples." << std::endl;
    std::cout << "   This demonstrates how neural networks scale to real-world problems." << std::endl;
    std::cout << "   More data generally leads to better generalization." << std::endl;
    std::cout << std::endl;

    // Performance evaluation
    if (final_test_accuracy >= 0.85) {
        std::cout << "✓ SUCCESS: The network achieved " << std::fixed << std::setprecision(1)
                  << (final_test_accuracy * 100) << "% accuracy!" << std::endl;
        std::cout << "  This exceeds the target of 85% accuracy on the test set." << std::endl;
        std::cout << "  The network has successfully learned to recognize handwritten digits." << std::endl;
    } else {
        std::cout << "⚠ PARTIAL SUCCESS: The network achieved " << std::fixed << std::setprecision(1)
                  << (final_test_accuracy * 100) << "% accuracy." << std::endl;
        std::cout << "  This is below the target of 85% accuracy." << std::endl;
        std::cout << "  To improve performance, consider:" << std::endl;
        std::cout << "  - Training for more epochs (try 20-30 epochs)" << std::endl;
        std::cout << "  - Using a larger training subset (try 30,000-60,000 images)" << std::endl;
        std::cout << "  - Adjusting the learning rate (try 0.005 or 0.02)" << std::endl;
        std::cout << "  - Adding more neurons to hidden layers" << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
