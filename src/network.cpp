#include "network.h"
#include "loss.h"
#include "training_monitor.h"
#include <stdexcept>
#include <algorithm>
#include <cmath>

/**
 * Construct an empty network.
 * Layers must be added using addLayer().
 */
Network::Network() : layers_() {
    // Empty network, layers will be added via addLayer()
}

/**
 * Add a layer to the network.
 *
 * Validates layer connectivity and neuron counts according to requirements:
 * - Requirement 1.2: Validate neuron count > 0
 * - Requirement 1.4: Automatically connect layers in sequence
 *
 * @param input_size Number of inputs to this layer
 * @param output_size Number of outputs (neurons) in this layer
 * @param activation Shared pointer to activation function
 * @throws std::invalid_argument if validation fails
 */
void Network::addLayer(size_t input_size, size_t output_size,
                       std::shared_ptr<ActivationFunction> activation) {
    // Requirement 1.2: Validate neuron count > 0
    if (input_size == 0) {
        throw std::invalid_argument(
            "Network addLayer: input_size must be greater than zero"
        );
    }
    if (output_size == 0) {
        throw std::invalid_argument(
            "Network addLayer: output_size must be greater than zero"
        );
    }

    // Validate activation function
    if (!activation) {
        throw std::invalid_argument(
            "Network addLayer: activation function cannot be null"
        );
    }

    // Requirement 1.4: Validate layer connectivity
    // If this is not the first layer, ensure input_size matches previous layer's output_size
    if (!layers_.empty()) {
        size_t prev_output_size = layers_.back().outputSize();
        if (input_size != prev_output_size) {
            throw std::invalid_argument(
                "Network addLayer: layer connectivity validation failed. "
                "Previous layer output size is " + std::to_string(prev_output_size) +
                ", but new layer input size is " + std::to_string(input_size) +
                ". Layer input size must match previous layer output size."
            );
        }
    }

    // Create and add the layer
    layers_.emplace_back(input_size, output_size, activation);
}

/**
 * Perform forward propagation through the entire network.
 *
 * Implements:
 * - Requirement 4.1: Execute forward pass through all layers
 * - Requirement 4.5: Return output of final layer
 * - Requirement 4.6: Validate input dimensions match network input layer
 *
 * @param input Input vector (size must match network input layer)
 * @return Output vector from final layer
 * @throws std::invalid_argument if validation fails
 */
Vector Network::predict(const Vector& input) {
    // Validate network has at least one layer
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network predict: cannot predict with empty network. "
            "Add at least one layer using addLayer()."
        );
    }

    // Requirement 4.6: Validate input dimensions match network input layer
    size_t expected_input_size = layers_[0].inputSize();
    if (input.size() != expected_input_size) {
        throw std::invalid_argument(
            "Network predict: input size " + std::to_string(input.size()) +
            " does not match network input size " + std::to_string(expected_input_size)
        );
    }

    // Requirement 4.1: Execute forward pass through all layers
    Vector current_output = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        current_output = layers_[i].forward(current_output);
    }

    // Requirement 4.5: Return output of final layer
    return current_output;
}

/**
 * Get the number of layers in the network.
 *
 * Implements Requirement 1.5: Provide methods to query network structure
 *
 * @return Number of layers
 */
size_t Network::numLayers() const {
    return layers_.size();
}

/**
 * Get the complete network topology.
 *
 * Implements:
 * - Requirement 1.3: Store complete topology (input, hidden, output dimensions)
 * - Requirement 1.5: Provide methods to query network structure
 *
 * Returns a vector containing the size of each layer:
 * [input_size_of_layer_0, output_size_of_layer_0, output_size_of_layer_1, ...]
 *
 * For a network with layers [2->4, 4->3, 3->1], returns [2, 4, 3, 1]
 *
 * @return Vector of layer sizes representing the complete topology
 */
std::vector<size_t> Network::getTopology() const {
    if (layers_.empty()) {
        return {};
    }

    std::vector<size_t> topology;

    // Add input size of first layer
    topology.push_back(layers_[0].inputSize());

    // Add output size of each layer
    for (const auto& layer : layers_) {
        topology.push_back(layer.outputSize());
    }

    return topology;
}

/**
 * Get a reference to a specific layer for weight initialization or inspection.
 *
 * @param index Layer index (0-based)
 * @return Reference to the layer
 * @throws std::out_of_range if index is invalid
 */
Layer& Network::getLayer(size_t index) {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Network getLayer: index " + std::to_string(index) +
            " is out of range (network has " + std::to_string(layers_.size()) + " layers)"
        );
    }
    return layers_[index];
}

/**
 * Get a const reference to a specific layer for inspection.
 *
 * @param index Layer index (0-based)
 * @return Const reference to the layer
 * @throws std::out_of_range if index is invalid
 */
const Layer& Network::getLayer(size_t index) const {
    if (index >= layers_.size()) {
        throw std::out_of_range(
            "Network getLayer: index " + std::to_string(index) +
            " is out of range (network has " + std::to_string(layers_.size()) + " layers)"
        );
    }
    return layers_[index];
}

/**
 * Perform backpropagation to compute gradients for all weights and biases.
 *
 * Implements the backpropagation algorithm according to Requirements 6.1-6.6:
 *
 * Algorithm:
 * 1. Compute output layer delta: δ^L = ∇_a L ⊙ σ'(z^L)
 *    where ∇_a L is the loss gradient and σ' is the activation derivative
 *
 * 2. For each hidden layer (backward from L-1 to 0):
 *    δ^l = (W^(l+1))^T * δ^(l+1) ⊙ σ'(z^l)
 *
 * 3. Compute weight gradients: ∂L/∂W^l = δ^l * (a^(l-1))^T
 *    where a^(l-1) is the input to layer l (output of layer l-1)
 *
 * 4. Compute bias gradients: ∂L/∂b^l = δ^l
 *
 * @param target Target output vector
 * @param loss_function Loss function to compute initial gradient
 * @param weight_gradients Output vector to store weight gradients (one matrix per layer)
 * @param bias_gradients Output vector to store bias gradients (one vector per layer)
 */
void Network::backpropagate(const Vector& target,
                            LossFunction& loss_function,
                            std::vector<Matrix>& weight_gradients,
                            std::vector<Vector>& bias_gradients) {
    // Validate network has layers
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network backpropagate: cannot backpropagate with empty network"
        );
    }

    size_t num_layers = layers_.size();

    // Initialize gradient storage
    weight_gradients.clear();
    bias_gradients.clear();

    // Storage for deltas (one per layer) - we'll build this backward
    std::vector<Vector> deltas;
    deltas.reserve(num_layers);

    // Step 1: Compute output layer delta
    // δ^L = ∇_a L ⊙ σ'(z^L)
    // Requirement 6.1, 6.5: Use loss gradient and activation derivative
    const Layer& output_layer = layers_[num_layers - 1];
    const Vector& output = output_layer.getLastOutput();
    const Vector& z_output = output_layer.getLastWeightedSum();  // weighted sum before activation

    // Get loss gradient: ∂L/∂a^L
    Vector loss_grad = loss_function.gradient(output, target);

    // Apply activation derivative element-wise: δ^L = ∇_a L ⊙ σ'(z^L)
    // Requirement 6.5: Apply activation function derivative
    Vector output_delta(output.size());
    std::shared_ptr<ActivationFunction> output_activation = output_layer.getActivation();
    for (size_t i = 0; i < output.size(); ++i) {
        double activation_deriv = output_activation->derivative(z_output[i]);
        output_delta[i] = loss_grad[i] * activation_deriv;
    }

    // Store output delta at the end (we'll build deltas in reverse order)
    deltas.push_back(output_delta);

    // Step 2: Propagate deltas backward through hidden layers
    // δ^l = (W^(l+1))^T * δ^(l+1) ⊙ σ'(z^l)
    // Requirement 6.3: Propagate gradients backward using chain rule
    for (int l = num_layers - 2; l >= 0; --l) {
        const Layer& current_layer = layers_[l];
        const Layer& next_layer = layers_[l + 1];
        const Vector& z_current = current_layer.getLastWeightedSum();

        // Get the delta from the next layer (which is at index 0 in our reverse-built vector)
        const Vector& next_delta = deltas[num_layers - 2 - l];

        // Compute (W^(l+1))^T * δ^(l+1)
        Matrix weights_next_transposed = next_layer.getWeights().transpose();
        Vector backprop_error = weights_next_transposed * next_delta;

        // Apply activation derivative element-wise: ⊙ σ'(z^l)
        // Requirement 6.5: Apply activation function derivative
        Vector current_delta(current_layer.outputSize());
        std::shared_ptr<ActivationFunction> current_activation = current_layer.getActivation();
        for (size_t i = 0; i < current_layer.outputSize(); ++i) {
            double activation_deriv = current_activation->derivative(z_current[i]);
            current_delta[i] = backprop_error[i] * activation_deriv;
        }

        deltas.push_back(current_delta);
    }

    // Now reverse deltas so they're in forward order (layer 0 to layer L)
    std::reverse(deltas.begin(), deltas.end());

    // Step 3 & 4: Compute weight and bias gradients for all layers
    // Requirement 6.1, 6.2: Compute gradients for all weights and biases
    for (size_t l = 0; l < num_layers; ++l) {
        const Layer& layer = layers_[l];
        const Vector& layer_input = layer.getLastInput();

        // Requirement 6.4: Use intermediate outputs from forward pass

        // Compute weight gradients: ∂L/∂W^l = δ^l * (a^(l-1))^T
        // This is an outer product: each element (i,j) = δ^l[i] * a^(l-1)[j]
        size_t output_size = layer.outputSize();
        size_t input_size = layer.inputSize();
        Matrix weight_grad(output_size, input_size);

        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                weight_grad(i, j) = deltas[l][i] * layer_input[j];
            }
        }

        weight_gradients.push_back(weight_grad);

        // Compute bias gradients: ∂L/∂b^l = δ^l
        bias_gradients.push_back(deltas[l]);
    }

    // Requirement 6.6: All gradients are now stored in weight_gradients and bias_gradients
}

/**
 * Train the network on a dataset using backpropagation with batch support.
 *
 * Implements training loop with configurable batch size according to Requirements 8.1-8.10.
 *
 * Batch Training Modes:
 * - batch_size = 1: Stochastic Gradient Descent (SGD)
 *   Updates parameters after each example
 *
 * - batch_size = 32 (typical): Mini-batch Gradient Descent
 *   Accumulates gradients over 32 examples, averages them, then updates
 *
 * - batch_size = dataset_size: Batch Gradient Descent
 *   Accumulates gradients over entire dataset, averages them, then updates once per epoch
 *
 * Algorithm:
 * For each epoch:
 *   Initialize epoch loss accumulator
 *   For each batch in dataset:
 *     Initialize gradient accumulators (zero matrices/vectors)
 *     For each example in batch:
 *       1. Forward pass: output = predict(input)
 *       2. Compute loss: L = loss_function.compute(output, target)
 *       3. Backpropagation: compute gradients for this example
 *       4. Accumulate gradients
 *     Average accumulated gradients: gradient_avg = gradient_sum / batch_size
 *     Update parameters: θ := θ - η * gradient_avg
 *   Compute average loss for epoch: epoch_loss / num_examples
 *   If monitor provided: record epoch metrics and print progress
 *
 * @param inputs Vector of input vectors (training examples)
 * @param targets Vector of target output vectors (labels)
 * @param epochs Number of complete passes through the dataset
 * @param learning_rate Learning rate (η) for gradient descent
 * @param loss_function Loss function to measure prediction error
 * @param batch_size Number of examples per batch (default = 1 for SGD)
 * @param monitor Optional TrainingMonitor to track and display progress (default = nullptr)
 * @return Vector of average loss values (one per epoch)
 * @throws std::invalid_argument if validation fails
 */
std::vector<double> Network::train(const std::vector<Vector>& inputs,
                                   const std::vector<Vector>& targets,
                                   size_t epochs,
                                   double learning_rate,
                                   LossFunction& loss_function,
                                   size_t batch_size,
                                   TrainingMonitor* monitor) {
    // Requirement 8.1: Validate training dataset
    if (inputs.empty()) {
        throw std::invalid_argument(
            "Network train: inputs cannot be empty"
        );
    }

    if (inputs.size() != targets.size()) {
        throw std::invalid_argument(
            "Network train: inputs size " + std::to_string(inputs.size()) +
            " does not match targets size " + std::to_string(targets.size())
        );
    }

    // Validate network has layers
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network train: cannot train empty network. Add layers first."
        );
    }

    // Validate epochs
    if (epochs == 0) {
        throw std::invalid_argument(
            "Network train: epochs must be greater than zero"
        );
    }

    // Validate learning rate
    if (learning_rate <= 0.0) {
        throw std::invalid_argument(
            "Network train: learning_rate must be positive, got " +
            std::to_string(learning_rate)
        );
    }

    // Requirement 8.10: Validate batch_size
    size_t dataset_size = inputs.size();
    if (batch_size == 0) {
        throw std::invalid_argument(
            "Network train: batch_size must be greater than zero"
        );
    }

    if (batch_size > dataset_size) {
        throw std::invalid_argument(
            "Network train: batch_size " + std::to_string(batch_size) +
            " cannot be greater than dataset size " + std::to_string(dataset_size)
        );
    }

    // Storage for loss history (one value per epoch)
    std::vector<double> loss_history;
    loss_history.reserve(epochs);

    // Requirement 8.2: Iterate through dataset for specified number of epochs
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double epoch_loss = 0.0;
        size_t num_batches = (dataset_size + batch_size - 1) / batch_size;  // ceiling division

        // Process dataset in batches
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            // Determine batch boundaries
            size_t batch_start = batch_idx * batch_size;
            size_t batch_end = std::min(batch_start + batch_size, dataset_size);
            size_t current_batch_size = batch_end - batch_start;

            // Initialize gradient accumulators (zero-initialized)
            std::vector<Matrix> accumulated_weight_gradients;
            std::vector<Vector> accumulated_bias_gradients;

            // Initialize accumulators with correct dimensions (all zeros)
            for (size_t l = 0; l < layers_.size(); ++l) {
                const Layer& layer = layers_[l];
                accumulated_weight_gradients.emplace_back(
                    layer.outputSize(), layer.inputSize(), 0.0
                );
                accumulated_bias_gradients.emplace_back(
                    layer.outputSize(), 0.0
                );
            }

            // Requirement 8.5: Accumulate gradients over batch
            for (size_t i = batch_start; i < batch_end; ++i) {
                // 1. Forward pass
                Vector output = predict(inputs[i]);

                // 2. Compute loss
                double loss = loss_function.compute(output, targets[i]);
                epoch_loss += loss;

                // 3. Backpropagation: compute gradients for this example
                std::vector<Matrix> example_weight_gradients;
                std::vector<Vector> example_bias_gradients;
                backpropagate(targets[i], loss_function,
                            example_weight_gradients, example_bias_gradients);

                // 4. Accumulate gradients
                for (size_t l = 0; l < layers_.size(); ++l) {
                    // Accumulate weight gradients
                    for (size_t row = 0; row < accumulated_weight_gradients[l].rows(); ++row) {
                        for (size_t col = 0; col < accumulated_weight_gradients[l].cols(); ++col) {
                            accumulated_weight_gradients[l](row, col) +=
                                example_weight_gradients[l](row, col);
                        }
                    }

                    // Accumulate bias gradients
                    for (size_t j = 0; j < accumulated_bias_gradients[l].size(); ++j) {
                        accumulated_bias_gradients[l][j] += example_bias_gradients[l][j];
                    }
                }
            }

            // Requirement 8.5: Average accumulated gradients over batch size
            // (For batch_size = 1, this is just the single example's gradient)
            for (size_t l = 0; l < layers_.size(); ++l) {
                // Average weight gradients
                for (size_t row = 0; row < accumulated_weight_gradients[l].rows(); ++row) {
                    for (size_t col = 0; col < accumulated_weight_gradients[l].cols(); ++col) {
                        accumulated_weight_gradients[l](row, col) /= current_batch_size;
                    }
                }

                // Average bias gradients
                for (size_t j = 0; j < accumulated_bias_gradients[l].size(); ++j) {
                    accumulated_bias_gradients[l][j] /= current_batch_size;
                }
            }

            // Update parameters using averaged gradients
            // Requirement 8.4: When batch_size = 1, this updates after each example (SGD)
            // Requirement 8.6: When batch_size = dataset_size, this updates once per epoch
            updateParameters(accumulated_weight_gradients,
                           accumulated_bias_gradients,
                           learning_rate);
        }

        // Requirement 8.7: Calculate and store average loss per epoch
        double avg_epoch_loss = epoch_loss / dataset_size;
        loss_history.push_back(avg_epoch_loss);

        // Requirement 8.8, 8.9: If monitor provided, record metrics and print progress
        if (monitor) {
            // Calculate accuracy for this epoch
            size_t correct_predictions = 0;
            for (size_t i = 0; i < dataset_size; ++i) {
                Vector output = predict(inputs[i]);

                // For classification: find index of max value in output and target
                // If they match, prediction is correct
                size_t predicted_class = 0;
                size_t target_class = 0;
                double max_output = output[0];
                double max_target = targets[i][0];

                for (size_t j = 1; j < output.size(); ++j) {
                    if (output[j] > max_output) {
                        max_output = output[j];
                        predicted_class = j;
                    }
                    if (targets[i][j] > max_target) {
                        max_target = targets[i][j];
                        target_class = j;
                    }
                }

                if (predicted_class == target_class) {
                    correct_predictions++;
                }
            }

            double accuracy = static_cast<double>(correct_predictions) / dataset_size;

            // Record epoch metrics
            monitor->recordEpoch(epoch, avg_epoch_loss, accuracy);

            // Print progress to user
            monitor->printProgress(epoch, epochs);
        }
    }

    return loss_history;
}

/**
 * Update network parameters using computed gradients.
 *
 * Implements gradient descent parameter update according to Requirements 7.1-7.5:
 *
 * Algorithm:
 * For each layer l:
 *   For each weight W^l[i][j]:
 *     W^l[i][j] := W^l[i][j] - η * ∂L/∂W^l[i][j]
 *   For each bias b^l[i]:
 *     b^l[i] := b^l[i] - η * ∂L/∂b^l[i]
 *
 * where η is the learning rate.
 *
 * Requirements validated:
 * - 7.1: Update all weights using gradient descent
 * - 7.2: Update all biases using gradient descent
 * - 7.3: Apply learning rate to parameter updates
 * - 7.5: Execute update according to formula: param_new = param_old - learning_rate * gradient
 *
 * @param weight_gradients Gradients for weights (one matrix per layer)
 * @param bias_gradients Gradients for biases (one vector per layer)
 * @param learning_rate Learning rate (η) to scale gradient updates
 */
void Network::updateParameters(const std::vector<Matrix>& weight_gradients,
                               const std::vector<Vector>& bias_gradients,
                               double learning_rate) {
    // Validate gradient dimensions match network structure
    if (weight_gradients.size() != layers_.size()) {
        throw std::invalid_argument(
            "Network updateParameters: weight_gradients size " +
            std::to_string(weight_gradients.size()) +
            " does not match number of layers " + std::to_string(layers_.size())
        );
    }

    if (bias_gradients.size() != layers_.size()) {
        throw std::invalid_argument(
            "Network updateParameters: bias_gradients size " +
            std::to_string(bias_gradients.size()) +
            " does not match number of layers " + std::to_string(layers_.size())
        );
    }

    // Requirement 7.1, 7.2, 7.3, 7.5: Update all weights and biases using gradient descent
    // Apply formula: θ_new = θ_old - η * gradient
    for (size_t l = 0; l < layers_.size(); ++l) {
        Layer& layer = layers_[l];
        Matrix& weights = layer.getWeights();
        Vector& biases = layer.getBiases();

        const Matrix& weight_grad = weight_gradients[l];
        const Vector& bias_grad = bias_gradients[l];

        // Validate gradient dimensions match layer dimensions
        if (weight_grad.rows() != weights.rows() || weight_grad.cols() != weights.cols()) {
            throw std::invalid_argument(
                "Network updateParameters: weight gradient dimensions (" +
                std::to_string(weight_grad.rows()) + "x" + std::to_string(weight_grad.cols()) +
                ") do not match layer " + std::to_string(l) + " weight dimensions (" +
                std::to_string(weights.rows()) + "x" + std::to_string(weights.cols()) + ")"
            );
        }

        if (bias_grad.size() != biases.size()) {
            throw std::invalid_argument(
                "Network updateParameters: bias gradient size " +
                std::to_string(bias_grad.size()) +
                " does not match layer " + std::to_string(l) + " bias size " +
                std::to_string(biases.size())
            );
        }

        // Update weights: W^l[i][j] := W^l[i][j] - η * ∂L/∂W^l[i][j]
        // Requirement 7.1, 7.3, 7.5
        for (size_t i = 0; i < weights.rows(); ++i) {
            for (size_t j = 0; j < weights.cols(); ++j) {
                weights(i, j) = weights(i, j) - learning_rate * weight_grad(i, j);
            }
        }

        // Update biases: b^l[i] := b^l[i] - η * ∂L/∂b^l[i]
        // Requirement 7.2, 7.3, 7.5
        for (size_t i = 0; i < biases.size(); ++i) {
            biases[i] = biases[i] - learning_rate * bias_grad[i];
        }
    }
}

/**
 * Validate the network on a test dataset without updating parameters.
 *
 * Performs forward pass on all test examples and computes average loss.
 * This method is const, ensuring no network parameters are modified.
 *
 * Implements:
 * - Requirement 11.2: Execute forward pass on all test examples without updating parameters
 * - Requirement 11.3: Calculate performance metrics (average loss)
 *
 * Algorithm:
 * 1. Validate inputs (non-empty, matching sizes)
 * 2. For each test example:
 *    a. Forward pass: output = predict(input)
 *    b. Compute loss: L = loss_function.compute(output, target)
 *    c. Accumulate loss
 * 3. Return average loss: total_loss / num_examples
 *
 * @param test_inputs Vector of test input vectors
 * @param test_targets Vector of test target output vectors
 * @param loss_function Loss function to measure prediction error
 * @return Average loss over the test dataset
 * @throws std::invalid_argument if validation fails
 */
double Network::validate(const std::vector<Vector>& test_inputs,
                        const std::vector<Vector>& test_targets,
                        LossFunction& loss_function) const {
    // Validate test dataset
    if (test_inputs.empty()) {
        throw std::invalid_argument(
            "Network validate: test_inputs cannot be empty"
        );
    }

    if (test_inputs.size() != test_targets.size()) {
        throw std::invalid_argument(
            "Network validate: test_inputs size " + std::to_string(test_inputs.size()) +
            " does not match test_targets size " + std::to_string(test_targets.size())
        );
    }

    // Validate network has layers
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network validate: cannot validate with empty network"
        );
    }

    // Accumulate loss over all test examples
    double total_loss = 0.0;
    size_t num_examples = test_inputs.size();

    // Requirement 11.2: Execute forward pass without updating parameters
    // (This method is const, so parameters cannot be modified)
    for (size_t i = 0; i < num_examples; ++i) {
        // Forward pass to get prediction
        // Note: We need to cast away const temporarily for predict()
        // This is safe because predict() doesn't modify parameters, only cached values
        Vector output = const_cast<Network*>(this)->predict(test_inputs[i]);

        // Compute loss for this example
        double loss = loss_function.compute(output, test_targets[i]);
        total_loss += loss;
    }

    // Requirement 11.3: Calculate and return average loss
    double average_loss = total_loss / num_examples;
    return average_loss;
}

/**
 * Calculate accuracy on a test dataset.
 *
 * For classification tasks (multi-class outputs), uses argmax comparison:
 * - A prediction is correct if argmax(predicted) == argmax(target)
 *
 * For regression/binary tasks (when threshold is used):
 * - A prediction is correct if all outputs are within threshold of target
 *
 * This method is const, ensuring no network parameters are modified.
 *
 * Implements:
 * - Requirement 11.1: Provide method to calculate accuracy on test dataset
 * - Requirement 11.2: Execute forward pass without updating parameters
 * - Requirement 11.3: Calculate performance metrics (accuracy)
 * - Requirement 11.5: Compare output with target using configurable threshold
 *
 * Algorithm:
 * 1. Validate inputs (non-empty, matching sizes)
 * 2. For each test example:
 *    a. Forward pass: output = predict(input)
 *    b. Determine if prediction is correct:
 *       - For multi-class: argmax(output) == argmax(target)
 *       - For regression: |output[i] - target[i]| <= threshold for all i
 *    c. Count correct predictions
 * 3. Return accuracy: correct_predictions / num_examples
 *
 * @param test_inputs Vector of test input vectors
 * @param test_targets Vector of test target output vectors
 * @param threshold Threshold for determining correctness (default = 0.5)
 *                  For classification: not used (uses argmax comparison)
 *                  For regression: maximum allowed error per output
 * @return Fraction of correct predictions (0.0 to 1.0)
 * @throws std::invalid_argument if validation fails
 */
double Network::calculateAccuracy(const std::vector<Vector>& test_inputs,
                                 const std::vector<Vector>& test_targets,
                                 double threshold) const {
    // Validate test dataset
    if (test_inputs.empty()) {
        throw std::invalid_argument(
            "Network calculateAccuracy: test_inputs cannot be empty"
        );
    }

    if (test_inputs.size() != test_targets.size()) {
        throw std::invalid_argument(
            "Network calculateAccuracy: test_inputs size " + std::to_string(test_inputs.size()) +
            " does not match test_targets size " + std::to_string(test_targets.size())
        );
    }

    // Validate network has layers
    if (layers_.empty()) {
        throw std::invalid_argument(
            "Network calculateAccuracy: cannot calculate accuracy with empty network"
        );
    }

    // Validate threshold is non-negative
    if (threshold < 0.0) {
        throw std::invalid_argument(
            "Network calculateAccuracy: threshold must be non-negative, got " +
            std::to_string(threshold)
        );
    }

    size_t correct_predictions = 0;
    size_t num_examples = test_inputs.size();

    // Requirement 11.2: Execute forward pass without updating parameters
    // (This method is const, so parameters cannot be modified)
    for (size_t i = 0; i < num_examples; ++i) {
        // Forward pass to get prediction
        // Note: We need to cast away const temporarily for predict()
        // This is safe because predict() doesn't modify parameters, only cached values
        Vector output = const_cast<Network*>(this)->predict(test_inputs[i]);
        const Vector& target = test_targets[i];

        // Validate output and target have same size
        if (output.size() != target.size()) {
            throw std::invalid_argument(
                "Network calculateAccuracy: output size " + std::to_string(output.size()) +
                " does not match target size " + std::to_string(target.size()) +
                " for example " + std::to_string(i)
            );
        }

        // Requirement 11.5: Compare output with target using threshold
        // Determine if prediction is correct based on output size
        bool is_correct = false;

        if (output.size() == 1) {
            // Binary/regression case: use threshold
            // Correct if |predicted - target| <= threshold
            double error = std::abs(output[0] - target[0]);
            is_correct = (error <= threshold);
        } else {
            // Multi-class classification case: use argmax comparison
            // Find index of maximum value in output and target
            size_t predicted_class = 0;
            size_t target_class = 0;
            double max_output = output[0];
            double max_target = target[0];

            for (size_t j = 1; j < output.size(); ++j) {
                if (output[j] > max_output) {
                    max_output = output[j];
                    predicted_class = j;
                }
                if (target[j] > max_target) {
                    max_target = target[j];
                    target_class = j;
                }
            }

            // Correct if predicted class matches target class
            is_correct = (predicted_class == target_class);
        }

        if (is_correct) {
            correct_predictions++;
        }
    }

    // Requirement 11.1, 11.3: Calculate and return accuracy as fraction of correct predictions
    double accuracy = static_cast<double>(correct_predictions) / num_examples;
    return accuracy;
}
