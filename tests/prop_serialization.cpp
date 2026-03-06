#include "network.h"
#include "serializer.h"
#include "activation.h"
#include "vector.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <sstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <memory>

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

// Generator for activation functions
rc::Gen<std::shared_ptr<ActivationFunction>> arbActivation() {
    return rc::gen::apply([](int choice) -> std::shared_ptr<ActivationFunction> {
        switch (choice % 3) {
            case 0: return std::make_shared<Sigmoid>();
            case 1: return std::make_shared<Tanh>();
            default: return std::make_shared<ReLU>();
        }
    }, rc::gen::inRange(0, 3));
}

// Generator for valid network topologies
// Returns a vector of layer sizes [input, hidden1, ..., hiddenN, output]
rc::Gen<std::vector<size_t>> arbNetworkTopology() {
    return rc::gen::apply([](size_t num_layers, std::vector<size_t> sizes) {
        // Ensure we have at least 2 layers (input->output)
        size_t actual_layers = std::max(size_t(2), num_layers);
        std::vector<size_t> topology;
        topology.reserve(actual_layers + 1);

        // Generate layer sizes (all must be > 0)
        for (size_t i = 0; i <= actual_layers; ++i) {
            size_t size = (i < sizes.size()) ? sizes[i] : 1;
            // Ensure size is at least 1 and at most 10 for reasonable test performance
            topology.push_back(std::max(size_t(1), std::min(size_t(10), size)));
        }

        return topology;
    }, rc::gen::inRange<size_t>(2, 5),  // 2-5 layers
       rc::gen::container<std::vector<size_t>>(6, rc::gen::inRange<size_t>(1, 10)));
}

// Generator for a complete network with random architecture and weights
rc::Gen<Network> arbNetwork() {
    return rc::gen::apply([](std::vector<size_t> topology, std::vector<std::shared_ptr<ActivationFunction>> activations) {
        Network net;

        // Add layers based on topology
        for (size_t i = 0; i < topology.size() - 1; ++i) {
            size_t input_size = topology[i];
            size_t output_size = topology[i + 1];
            auto activation = (i < activations.size()) ? activations[i] : std::make_shared<Sigmoid>();
            net.addLayer(input_size, output_size, activation);
        }

        return net;
    }, arbNetworkTopology(),
       rc::gen::container<std::vector<std::shared_ptr<ActivationFunction>>>(5, arbActivation()));
}

// **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.9**
// Feature: neural-network-framework, Property 27: Serialization Round-Trip
// For any valid network, serializing then deserializing then serializing again
// should produce identical output
RC_GTEST_PROP(SerializationPropertyTest, RoundTripProperty, ()) {
    // Generate a random network
    auto network = *arbNetwork();

    // First serialization
    std::stringstream ss1;
    Serializer::serialize(network, ss1);
    std::string first_serialization = ss1.str();

    // Deserialize
    std::stringstream ss2(first_serialization);
    Network deserialized = Serializer::deserialize(ss2);

    // Second serialization
    std::stringstream ss3;
    Serializer::serialize(deserialized, ss3);
    std::string second_serialization = ss3.str();

    // Verify the two serializations are identical
    RC_ASSERT(first_serialization == second_serialization);
}

// **Validates: Requirements 10.4**
// Feature: neural-network-framework, Property 28: Deserialization Completeness
// Reconstructed network should have identical topology, weights, biases,
// and activation functions
RC_GTEST_PROP(SerializationPropertyTest, DeserializationCompleteness, ()) {
    // Generate a random network
    auto original = *arbNetwork();

    // Serialize and deserialize
    std::stringstream ss;
    Serializer::serialize(original, ss);
    Network reconstructed = Serializer::deserialize(ss);

    // Verify topology
    RC_ASSERT(reconstructed.numLayers() == original.numLayers());

    std::vector<size_t> orig_topology = original.getTopology();
    std::vector<size_t> recon_topology = reconstructed.getTopology();
    RC_ASSERT(orig_topology.size() == recon_topology.size());

    for (size_t i = 0; i < orig_topology.size(); ++i) {
        RC_ASSERT(orig_topology[i] == recon_topology[i]);
    }

    // Verify each layer
    for (size_t i = 0; i < original.numLayers(); ++i) {
        const Layer& orig_layer = original.getLayer(i);
        const Layer& recon_layer = reconstructed.getLayer(i);

        // Verify activation function
        RC_ASSERT(orig_layer.activationName() == recon_layer.activationName());

        // Verify layer dimensions
        RC_ASSERT(orig_layer.inputSize() == recon_layer.inputSize());
        RC_ASSERT(orig_layer.outputSize() == recon_layer.outputSize());

        // Verify weights
        const Matrix& orig_weights = orig_layer.getWeights();
        const Matrix& recon_weights = recon_layer.getWeights();
        RC_ASSERT(orig_weights.rows() == recon_weights.rows());
        RC_ASSERT(orig_weights.cols() == recon_weights.cols());

        for (size_t r = 0; r < orig_weights.rows(); ++r) {
            for (size_t c = 0; c < orig_weights.cols(); ++c) {
                RC_ASSERT(approxEqual(orig_weights(r, c), recon_weights(r, c)));
            }
        }

        // Verify biases
        const Vector& orig_biases = orig_layer.getBiases();
        const Vector& recon_biases = recon_layer.getBiases();
        RC_ASSERT(orig_biases.size() == recon_biases.size());

        for (size_t j = 0; j < orig_biases.size(); ++j) {
            RC_ASSERT(approxEqual(orig_biases[j], recon_biases[j]));
        }
    }
}

// **Validates: Requirements 10.5, 10.6**
// Feature: neural-network-framework, Property 29: Corrupted File Detection
// Modified/corrupted files should be detected with descriptive errors
RC_GTEST_PROP(SerializationPropertyTest, CorruptedFileDetection, ()) {
    // Generate a random network
    auto network = *arbNetwork();

    // Serialize to string
    std::stringstream ss;
    Serializer::serialize(network, ss);
    std::string serialized = ss.str();

    // Generate a random corruption type
    auto corruption_type = *rc::gen::inRange(0, 5);

    std::string corrupted;

    switch (corruption_type) {
        case 0: {
            // Corrupt the header
            corrupted = serialized;
            size_t pos = corrupted.find("NEURAL_NETWORK_V1");
            if (pos != std::string::npos) {
                corrupted.replace(pos, 17, "INVALID_HEADER___");
            }
            break;
        }
        case 1: {
            // Remove the END marker
            corrupted = serialized;
            size_t pos = corrupted.find("END");
            if (pos != std::string::npos) {
                corrupted.erase(pos, 3);
            }
            break;
        }
        case 2: {
            // Corrupt an activation function name
            corrupted = serialized;
            size_t pos = corrupted.find("ACTIVATION");
            if (pos != std::string::npos) {
                size_t line_end = corrupted.find('\n', pos);
                if (line_end != std::string::npos) {
                    corrupted.replace(pos, line_end - pos, "ACTIVATION invalid_func");
                }
            }
            break;
        }
        case 3: {
            // Corrupt a dimension value
            corrupted = serialized;
            size_t pos = corrupted.find("INPUT_SIZE");
            if (pos != std::string::npos) {
                size_t line_end = corrupted.find('\n', pos);
                if (line_end != std::string::npos) {
                    corrupted.replace(pos, line_end - pos, "INPUT_SIZE -1");
                }
            }
            break;
        }
        case 4: {
            // Insert NaN into weights
            corrupted = serialized;
            size_t pos = corrupted.find("WEIGHTS");
            if (pos != std::string::npos) {
                // Find the first number after WEIGHTS line
                size_t newline = corrupted.find('\n', pos);
                if (newline != std::string::npos) {
                    size_t next_newline = corrupted.find('\n', newline + 1);
                    if (next_newline != std::string::npos) {
                        // Replace first weight value with NaN
                        size_t space_pos = corrupted.find(' ', newline);
                        if (space_pos != std::string::npos && space_pos < next_newline) {
                            corrupted.replace(newline + 1, space_pos - newline - 1, "nan");
                        }
                    }
                }
            }
            break;
        }
    }

    // Try to deserialize the corrupted data
    std::stringstream corrupted_ss(corrupted);
    bool caught_exception = false;

    try {
        Network loaded = Serializer::deserialize(corrupted_ss);
    } catch (const std::runtime_error& e) {
        caught_exception = true;
        // Verify error message is descriptive (contains some context)
        std::string msg = e.what();
        RC_ASSERT(!msg.empty());
    } catch (const std::invalid_argument& e) {
        caught_exception = true;
        // Verify error message is descriptive
        std::string msg = e.what();
        RC_ASSERT(!msg.empty());
    } catch (const std::exception& e) {
        // Any exception is acceptable for corrupted data
        caught_exception = true;
    }

    // Verify that an exception was thrown
    RC_ASSERT(caught_exception);
}

// Additional property: Predictions should be identical after deserialization
// This ensures functional equivalence, not just structural equivalence
RC_GTEST_PROP(SerializationPropertyTest, PredictionEquivalence, ()) {
    // Generate a random network
    auto network = *arbNetwork();

    // Generate a random input vector matching the network's input size
    std::vector<size_t> topology = network.getTopology();
    size_t input_size = topology[0];

    auto input_values = *rc::gen::container<std::vector<double>>(
        input_size,
        rc::gen::map(rc::gen::inRange(0, 10000), [](int i) {
            return -5.0 + 10.0 * (i / 10000.0);
        })
    );

    Vector input(input_size);
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = input_values[i];
    }

    // Get prediction from original network
    Vector original_output = network.predict(input);

    // Serialize and deserialize
    std::stringstream ss;
    Serializer::serialize(network, ss);
    Network deserialized = Serializer::deserialize(ss);

    // Get prediction from deserialized network
    Vector deserialized_output = deserialized.predict(input);

    // Verify outputs are identical
    RC_ASSERT(original_output.size() == deserialized_output.size());

    for (size_t i = 0; i < original_output.size(); ++i) {
        RC_ASSERT(approxEqual(original_output[i], deserialized_output[i]));
    }
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
