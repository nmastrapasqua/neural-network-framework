#include "network.h"
#include "activation.h"
#include "loss.h"
#include "training_monitor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <iomanip>
#include <random>
#include <algorithm>
#include <map>

/**
 * Iris Dataset Classification Example
 *
 * This example demonstrates multi-class classification on the famous Iris dataset.
 * The Iris dataset is one of the most well-known datasets in machine learning,
 * introduced by Ronald Fisher in 1936.
 *
 * Dataset Description:
 * - 150 samples total (50 per class)
 * - 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width (in cm)
 * - 3 classes: Iris-setosa, Iris-versicolor, Iris-virginica
 * - Task: Classify iris flowers based on their measurements
 *
 * Network Architecture:
 * - Input layer: 4 neurons (one per feature)
 * - Hidden layer: 8 neurons (tanh activation)
 * - Output layer: 3 neurons (sigmoid activation, one per class)
 *
 * Key Concepts Demonstrated:
 * - Classification on tabular/structured data (not images)
 * - Feature normalization (important for neural networks)
 * - Train/test split on small datasets
 * - Multi-class classification with one-hot encoding
 * - CSV file parsing
 *
 * Educational Value:
 * - Shows that neural networks work on any type of data
 * - Demonstrates importance of data preprocessing
 * - Small dataset makes training very fast (seconds)
 * - Easy to understand features (flower measurements)
 */

struct IrisData {
    std::vector<Vector> features;
    std::vector<Vector> labels;
    std::vector<std::string> class_names;
};

/**
 * Parse Iris CSV file and load data
 */
IrisData loadIrisDataset(const std::string& filename) {
    IrisData data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Map species names to class indices
    std::map<std::string, size_t> species_to_index;
    species_to_index["Iris-setosa"] = 0;
    species_to_index["Iris-versicolor"] = 1;
    species_to_index["Iris-virginica"] = 2;

    data.class_names = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};

    std::string line;
    // Skip header
    std::getline(file, line);

    // Read data lines
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;

        // Skip ID
        std::getline(ss, token, ',');

        // Read 4 features
        std::vector<double> features_vec;
        for (int i = 0; i < 4; ++i) {
            std::getline(ss, token, ',');
            features_vec.push_back(std::stod(token));
        }

        // Read species
        std::getline(ss, token, ',');

        // Create feature vector
        Vector feature_vector(4);
        for (size_t i = 0; i < 4; ++i) {
            feature_vector[i] = features_vec[i];
        }
        data.features.push_back(feature_vector);

        // Create one-hot encoded label
        size_t class_index = species_to_index[token];
        Vector label(3, 0.0);
        label[class_index] = 1.0;
        data.labels.push_back(label);
    }

    file.close();
    return data;
}

/**
 * Normalize features to [0, 1] range
 * This is important for neural network training
 */
void normalizeFeatures(std::vector<Vector>& features) {
    if (features.empty()) return;

    size_t num_features = features[0].size();

    // Find min and max for each feature
    std::vector<double> min_vals(num_features, std::numeric_limits<double>::max());
    std::vector<double> max_vals(num_features, std::numeric_limits<double>::lowest());

    for (const auto& feature : features) {
        for (size_t i = 0; i < num_features; ++i) {
            min_vals[i] = std::min(min_vals[i], feature[i]);
            max_vals[i] = std::max(max_vals[i], feature[i]);
        }
    }

    // Normalize: (x - min) / (max - min)
    for (auto& feature : features) {
        for (size_t i = 0; i < num_features; ++i) {
            double range = max_vals[i] - min_vals[i];
            if (range > 0) {
                feature[i] = (feature[i] - min_vals[i]) / range;
            }
        }
    }
}

/**
 * Shuffle dataset (for train/test split)
 */
void shuffleDataset(std::vector<Vector>& features, std::vector<Vector>& labels, unsigned int seed) {
    std::mt19937 rng(seed);

    // Create indices and shuffle them
    std::vector<size_t> indices(features.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::shuffle(indices.begin(), indices.end(), rng);

    // Reorder features and labels
    std::vector<Vector> shuffled_features;
    std::vector<Vector> shuffled_labels;

    for (size_t idx : indices) {
        shuffled_features.push_back(features[idx]);
        shuffled_labels.push_back(labels[idx]);
    }

    features = shuffled_features;
    labels = shuffled_labels;
}

/**
 * Split dataset into train and test sets
 */
void trainTestSplit(const std::vector<Vector>& features,
                   const std::vector<Vector>& labels,
                   std::vector<Vector>& train_features,
                   std::vector<Vector>& train_labels,
                   std::vector<Vector>& test_features,
                   std::vector<Vector>& test_labels,
                   double train_ratio) {
    size_t train_size = static_cast<size_t>(features.size() * train_ratio);

    train_features.clear();
    train_labels.clear();
    test_features.clear();
    test_labels.clear();

    for (size_t i = 0; i < features.size(); ++i) {
        if (i < train_size) {
            train_features.push_back(features[i]);
            train_labels.push_back(labels[i]);
        } else {
            test_features.push_back(features[i]);
            test_labels.push_back(labels[i]);
        }
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Iris Dataset Classification Demo" << std::endl;
    std::cout << "========================================" << std::endl << std::endl;

    // ============================================================================
    // STEP 1: Load Iris dataset
    // ============================================================================

    std::cout << "Step 1: Loading Iris dataset" << std::endl;
    std::cout << "----------------------------" << std::endl;

    IrisData data;
    try {
        data = loadIrisDataset("iris_dataset/iris.csv");
        std::cout << "Dataset loaded successfully!" << std::endl;
        std::cout << "  - Total samples: " << data.features.size() << std::endl;
        std::cout << "  - Features per sample: " << data.features[0].size() << std::endl;
        std::cout << "  - Classes: " << data.class_names.size() << std::endl;
        for (size_t i = 0; i < data.class_names.size(); ++i) {
            std::cout << "    " << i << ": " << data.class_names[i] << std::endl;
        }
        std::cout << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return 1;
    }

    // ============================================================================
    // STEP 2: Preprocess data
    // ============================================================================

    std::cout << "Step 2: Preprocessing data" << std::endl;
    std::cout << "--------------------------" << std::endl;

    // Show sample before normalization
    std::cout << "Sample before normalization:" << std::endl;
    std::cout << "  Features: [" << data.features[0][0] << ", "
              << data.features[0][1] << ", "
              << data.features[0][2] << ", "
              << data.features[0][3] << "]" << std::endl;

    // Normalize features to [0, 1]
    normalizeFeatures(data.features);

    std::cout << "Sample after normalization:" << std::endl;
    std::cout << "  Features: [" << std::fixed << std::setprecision(3)
              << data.features[0][0] << ", "
              << data.features[0][1] << ", "
              << data.features[0][2] << ", "
              << data.features[0][3] << "]" << std::endl;
    std::cout << std::endl;

    // Shuffle dataset
    std::cout << "Shuffling dataset with seed=123..." << std::endl;
    shuffleDataset(data.features, data.labels, 123);
    std::cout << std::endl;

    // ============================================================================
    // STEP 3: Split into train and test sets
    // ============================================================================

    std::cout << "Step 3: Splitting into train/test sets" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    std::vector<Vector> train_features, train_labels;
    std::vector<Vector> test_features, test_labels;

    const double train_ratio = 0.8;  // 80% train, 20% test
    trainTestSplit(data.features, data.labels,
                  train_features, train_labels,
                  test_features, test_labels,
                  train_ratio);

    std::cout << "Split ratio: " << (train_ratio * 100) << "% train, "
              << ((1 - train_ratio) * 100) << "% test" << std::endl;
    std::cout << "  - Training samples: " << train_features.size() << std::endl;
    std::cout << "  - Test samples: " << test_features.size() << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 4: Create neural network
    // ============================================================================

    std::cout << "Step 4: Creating neural network" << std::endl;
    std::cout << "--------------------------------" << std::endl;

    Network network;
    auto tanh_activation = std::make_shared<Tanh>();
    auto sigmoid_activation = std::make_shared<Sigmoid>();

    // Architecture: [4, 6, 3]
    // - 4 input features
    // - 6 hidden neurons (tanh) - balanced complexity
    // - 3 output classes (sigmoid)
    network.addLayer(4, 6, tanh_activation);
    network.addLayer(6, 3, sigmoid_activation);

    std::cout << "Network architecture: [4, 6, 3]" << std::endl;
    std::cout << "  - Input layer: 4 neurons (flower measurements)" << std::endl;
    std::cout << "  - Hidden layer: 6 neurons (tanh activation)" << std::endl;
    std::cout << "  - Output layer: 3 neurons (sigmoid activation, one per class)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 5: Initialize weights
    // ============================================================================

    std::cout << "Step 5: Initializing weights" << std::endl;
    std::cout << "-----------------------------" << std::endl;

    std::srand(123);
    network.getLayer(0).initializeXavier(4, 6);
    network.getLayer(1).initializeXavier(6, 3);

    std::cout << "Weights initialized with Xavier initialization" << std::endl;
    std::cout << "Random seed: 123 (for reproducibility)" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 6: Train the network
    // ============================================================================

    std::cout << "Step 6: Training the network" << std::endl;
    std::cout << "----------------------------" << std::endl;

    const size_t epochs = 300;
    const double learning_rate = 0.15;
    MeanSquaredError loss_function;

    std::cout << "Training configuration:" << std::endl;
    std::cout << "  - Epochs: " << epochs << std::endl;
    std::cout << "  - Learning rate: " << learning_rate << std::endl;
    std::cout << "  - Loss function: MSE" << std::endl;
    std::cout << "  - Batch size: 1 (SGD)" << std::endl;
    std::cout << std::endl;

    std::cout << "Training progress (every 100 epochs):" << std::endl;
    std::cout << "Epoch | Train Loss | Train Acc | Test Acc" << std::endl;
    std::cout << "------|------------|-----------|----------" << std::endl;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        // Train for one epoch
        std::vector<double> loss_history = network.train(
            train_features,
            train_labels,
            1,
            learning_rate,
            loss_function,
            1
        );

        // Display progress every 100 epochs
        if ((epoch + 1) % 100 == 0 || epoch == 0) {
            double train_acc = network.calculateAccuracy(train_features, train_labels);
            double test_acc = network.calculateAccuracy(test_features, test_labels);

            std::cout << std::setw(5) << (epoch + 1) << " | "
                      << std::fixed << std::setprecision(6) << loss_history[0] << " | "
                      << std::fixed << std::setprecision(2) << (train_acc * 100) << "%   | "
                      << std::fixed << std::setprecision(2) << (test_acc * 100) << "%" << std::endl;
        }
    }

    std::cout << std::endl;

    // ============================================================================
    // STEP 7: Final evaluation
    // ============================================================================

    std::cout << "Step 7: Final evaluation" << std::endl;
    std::cout << "------------------------" << std::endl;

    double final_train_acc = network.calculateAccuracy(train_features, train_labels);
    double final_test_acc = network.calculateAccuracy(test_features, test_labels);
    double final_test_loss = network.validate(test_features, test_labels, loss_function);

    std::cout << "Final performance:" << std::endl;
    std::cout << "  - Training accuracy: " << std::fixed << std::setprecision(2)
              << (final_train_acc * 100) << "%" << std::endl;
    std::cout << "  - Test accuracy: " << std::fixed << std::setprecision(2)
              << (final_test_acc * 100) << "%" << std::endl;
    std::cout << "  - Test loss: " << std::fixed << std::setprecision(6)
              << final_test_loss << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // STEP 8: Show predictions on test set
    // ============================================================================

    std::cout << "Step 8: Sample predictions on test set" << std::endl;
    std::cout << "---------------------------------------" << std::endl;

    std::cout << "Showing first 10 test predictions:" << std::endl;
    std::cout << std::endl;

    size_t num_show = std::min(size_t(10), test_features.size());
    int correct = 0;

    for (size_t i = 0; i < num_show; ++i) {
        Vector prediction = network.predict(test_features[i]);

        // Get predicted class (argmax)
        size_t predicted_class = 0;
        double max_prob = prediction[0];
        for (size_t j = 1; j < 3; ++j) {
            if (prediction[j] > max_prob) {
                max_prob = prediction[j];
                predicted_class = j;
            }
        }

        // Get actual class
        size_t actual_class = 0;
        for (size_t j = 0; j < 3; ++j) {
            if (test_labels[i][j] > 0.5) {
                actual_class = j;
                break;
            }
        }

        bool is_correct = (predicted_class == actual_class);
        if (is_correct) correct++;

        std::cout << "Sample " << (i + 1) << ":" << std::endl;
        std::cout << "  Actual: " << data.class_names[actual_class] << std::endl;
        std::cout << "  Predicted: " << data.class_names[predicted_class];
        std::cout << (is_correct ? " ✓" : " ✗") << std::endl;
        std::cout << "  Probabilities: [";
        for (size_t j = 0; j < 3; ++j) {
            std::cout << std::fixed << std::setprecision(3) << prediction[j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << std::endl;
    }

    std::cout << "Accuracy on shown samples: " << correct << "/" << num_show
              << " = " << std::fixed << std::setprecision(1)
              << (100.0 * correct / num_show) << "%" << std::endl;
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

    std::cout << "1. Tabular Data Classification:" << std::endl;
    std::cout << "   Neural networks work on any type of data, not just images." << std::endl;
    std::cout << "   Iris has 4 numerical features (flower measurements)." << std::endl;
    std::cout << std::endl;

    std::cout << "2. Feature Normalization:" << std::endl;
    std::cout << "   We normalized features to [0, 1] range before training." << std::endl;
    std::cout << "   This is crucial for neural networks to train effectively." << std::endl;
    std::cout << "   Without normalization, features with larger values dominate." << std::endl;
    std::cout << std::endl;

    std::cout << "3. Small Dataset Challenges:" << std::endl;
    std::cout << "   Iris has only 150 samples (120 train, 30 test)." << std::endl;
    std::cout << "   Small datasets are prone to overfitting." << std::endl;
    std::cout << "   We use a simple architecture (6 hidden neurons) to avoid this." << std::endl;
    std::cout << "   Fewer neurons = less capacity to memorize = better generalization." << std::endl;
    std::cout << std::endl;

    std::cout << "4. Multi-Class Classification:" << std::endl;
    std::cout << "   3 output neurons, one per class (one-hot encoding)." << std::endl;
    std::cout << "   Predicted class = neuron with highest activation." << std::endl;
    std::cout << std::endl;

    std::cout << "5. Classical ML Dataset:" << std::endl;
    std::cout << "   Iris is a benchmark dataset from 1936!" << std::endl;
    std::cout << "   It's often used to compare different ML algorithms." << std::endl;
    std::cout << "   Neural networks are actually overkill for Iris," << std::endl;
    std::cout << "   but it's great for learning and demonstration." << std::endl;
    std::cout << std::endl;

    if (final_test_acc >= 0.90) {
        std::cout << "✓ EXCELLENT: " << std::fixed << std::setprecision(1)
                  << (final_test_acc * 100) << "% test accuracy!" << std::endl;
        std::cout << "  The network successfully learned to classify iris flowers." << std::endl;
    } else if (final_test_acc >= 0.80) {
        std::cout << "✓ GOOD: " << std::fixed << std::setprecision(1)
                  << (final_test_acc * 100) << "% test accuracy." << std::endl;
        std::cout << "  The network learned reasonably well." << std::endl;
    } else {
        std::cout << "⚠ NEEDS IMPROVEMENT: " << std::fixed << std::setprecision(1)
                  << (final_test_acc * 100) << "% test accuracy." << std::endl;
        std::cout << "  Try more epochs or adjust learning rate." << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
