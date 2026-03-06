#include "serializer.h"
#include "network.h"
#include "activation.h"
#include <stdexcept>
#include <iomanip>
#include <cmath>
#include <memory>

void Serializer::serialize(const Network& network, std::ostream& out) {
    // Check if stream is good
    if (!out.good()) {
        throw std::runtime_error("Serialization failed: output stream is not in good state");
    }

    // Write header
    out << "NEURAL_NETWORK_V1" << std::endl;

    // Write number of layers
    size_t num_layers = network.numLayers();
    out << "LAYERS " << num_layers << std::endl;

    // Write each layer
    for (size_t i = 0; i < num_layers; ++i) {
        const Layer& layer = network.getLayer(i);

        out << "LAYER " << i << std::endl;

        // Write layer dimensions
        out << "  INPUT_SIZE " << layer.inputSize() << std::endl;
        out << "  OUTPUT_SIZE " << layer.outputSize() << std::endl;

        // Write activation function name
        out << "  ACTIVATION " << layer.activationName() << std::endl;

        // Write weights
        writeMatrix(layer.getWeights(), out, "WEIGHTS");

        // Write biases
        writeVector(layer.getBiases(), out, "BIASES");
    }

    // Write footer
    out << "END" << std::endl;

    // Check if writing succeeded
    if (!out.good()) {
        throw std::runtime_error("Serialization failed: error writing to output stream");
    }
}

void Serializer::writeMatrix(const Matrix& mat, std::ostream& out, const std::string& label) {
    // Write matrix header with dimensions
    out << "  " << label << " " << mat.rows() << " " << mat.cols() << std::endl;

    // Write matrix data row by row
    // Use fixed precision for consistent output
    out << std::fixed << std::setprecision(10);

    for (size_t i = 0; i < mat.rows(); ++i) {
        out << "    ";
        for (size_t j = 0; j < mat.cols(); ++j) {
            out << mat(i, j);
            if (j < mat.cols() - 1) {
                out << " ";
            }
        }
        out << std::endl;
    }

    // Reset to default formatting
    out << std::defaultfloat;
}

void Serializer::writeVector(const Vector& vec, std::ostream& out, const std::string& label) {
    // Write vector header with size
    out << "  " << label << " " << vec.size() << std::endl;

    // Write vector data on a single line
    // Use fixed precision for consistent output
    out << std::fixed << std::setprecision(10);

    out << "    ";
    for (size_t i = 0; i < vec.size(); ++i) {
        out << vec[i];
        if (i < vec.size() - 1) {
            out << " ";
        }
    }
    out << std::endl;

    // Reset to default formatting
    out << std::defaultfloat;
}

Network Serializer::deserialize(std::istream& in) {
    // Check if stream is good
    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: input stream is not in good state");
    }

    // Read and validate header
    expectKeyword(in, "NEURAL_NETWORK_V1");

    // Read number of layers
    expectKeyword(in, "LAYERS");
    size_t num_layers;
    in >> num_layers;

    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: could not read number of layers");
    }

    if (num_layers == 0) {
        throw std::invalid_argument("Deserialization failed: network must have at least one layer");
    }

    // Create empty network
    Network network;

    // Read each layer
    for (size_t i = 0; i < num_layers; ++i) {
        // Read layer header
        expectKeyword(in, "LAYER");
        size_t layer_index;
        in >> layer_index;

        if (!in.good()) {
            throw std::runtime_error("Deserialization failed: could not read layer index at layer " + std::to_string(i));
        }

        if (layer_index != i) {
            throw std::runtime_error("Deserialization failed: expected layer " + std::to_string(i) +
                                   ", found layer " + std::to_string(layer_index));
        }

        // Read input size
        expectKeyword(in, "INPUT_SIZE");
        size_t input_size;
        in >> input_size;

        if (!in.good()) {
            throw std::runtime_error("Deserialization failed: could not read input size for layer " + std::to_string(i));
        }

        if (input_size == 0) {
            throw std::invalid_argument("Deserialization failed: input size must be > 0 for layer " + std::to_string(i));
        }

        // Read output size
        expectKeyword(in, "OUTPUT_SIZE");
        size_t output_size;
        in >> output_size;

        if (!in.good()) {
            throw std::runtime_error("Deserialization failed: could not read output size for layer " + std::to_string(i));
        }

        if (output_size == 0) {
            throw std::invalid_argument("Deserialization failed: output size must be > 0 for layer " + std::to_string(i));
        }

        // Validate layer connectivity (except for first layer)
        if (i > 0) {
            size_t prev_output_size = network.getLayer(i - 1).outputSize();
            if (input_size != prev_output_size) {
                throw std::invalid_argument("Deserialization failed: layer " + std::to_string(i) +
                                          " input size (" + std::to_string(input_size) +
                                          ") does not match previous layer output size (" +
                                          std::to_string(prev_output_size) + ")");
            }
        }

        // Read activation function
        expectKeyword(in, "ACTIVATION");
        std::string activation_name;
        in >> activation_name;

        if (!in.good()) {
            throw std::runtime_error("Deserialization failed: could not read activation function for layer " + std::to_string(i));
        }

        std::shared_ptr<ActivationFunction> activation;
        try {
            activation = createActivation(activation_name);
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument("Deserialization failed: invalid activation function '" +
                                      activation_name + "' for layer " + std::to_string(i));
        }

        // Add layer to network
        network.addLayer(input_size, output_size, activation);

        // Read weights
        Matrix weights = readMatrix(in, "WEIGHTS");

        // Validate weights dimensions
        if (weights.rows() != output_size || weights.cols() != input_size) {
            throw std::invalid_argument("Deserialization failed: weights dimensions (" +
                                      std::to_string(weights.rows()) + "x" + std::to_string(weights.cols()) +
                                      ") do not match layer dimensions (" +
                                      std::to_string(output_size) + "x" + std::to_string(input_size) +
                                      ") for layer " + std::to_string(i));
        }

        // Validate all weight values are finite
        for (size_t r = 0; r < weights.rows(); ++r) {
            for (size_t c = 0; c < weights.cols(); ++c) {
                validateFinite(weights(r, c), "weight[" + std::to_string(r) + "][" +
                             std::to_string(c) + "] in layer " + std::to_string(i));
            }
        }

        // Read biases
        Vector biases = readVector(in, "BIASES");

        // Validate biases dimensions
        if (biases.size() != output_size) {
            throw std::invalid_argument("Deserialization failed: biases size (" +
                                      std::to_string(biases.size()) +
                                      ") does not match layer output size (" +
                                      std::to_string(output_size) + ") for layer " + std::to_string(i));
        }

        // Validate all bias values are finite
        for (size_t j = 0; j < biases.size(); ++j) {
            validateFinite(biases[j], "bias[" + std::to_string(j) + "] in layer " + std::to_string(i));
        }

        // Set weights and biases for the layer
        Layer& layer = network.getLayer(i);
        layer.getWeights() = weights;
        layer.getBiases() = biases;
    }

    // Read and validate footer
    expectKeyword(in, "END");

    return network;
}

Matrix Serializer::readMatrix(std::istream& in, const std::string& expected_label) {
    // Read label
    std::string label;
    in >> label;

    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: could not read matrix label (expected '" +
                               expected_label + "')");
    }

    if (label != expected_label) {
        throw std::runtime_error("Deserialization failed: expected '" + expected_label +
                               "', found '" + label + "'");
    }

    // Read dimensions
    size_t rows, cols;
    in >> rows >> cols;

    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: could not read matrix dimensions for '" +
                               expected_label + "'");
    }

    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Deserialization failed: matrix dimensions must be > 0 for '" +
                                  expected_label + "', got " + std::to_string(rows) + "x" +
                                  std::to_string(cols));
    }

    // Create matrix
    Matrix mat(rows, cols);

    // Read matrix data
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            double value;
            in >> value;

            if (!in.good()) {
                throw std::runtime_error("Deserialization failed: could not read matrix value at [" +
                                       std::to_string(i) + "][" + std::to_string(j) + "] for '" +
                                       expected_label + "'");
            }

            mat(i, j) = value;
        }
    }

    return mat;
}

Vector Serializer::readVector(std::istream& in, const std::string& expected_label) {
    // Read label
    std::string label;
    in >> label;

    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: could not read vector label (expected '" +
                               expected_label + "')");
    }

    if (label != expected_label) {
        throw std::runtime_error("Deserialization failed: expected '" + expected_label +
                               "', found '" + label + "'");
    }

    // Read size
    size_t size;
    in >> size;

    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: could not read vector size for '" +
                               expected_label + "'");
    }

    if (size == 0) {
        throw std::invalid_argument("Deserialization failed: vector size must be > 0 for '" +
                                  expected_label + "'");
    }

    // Create vector
    Vector vec(size);

    // Read vector data
    for (size_t i = 0; i < size; ++i) {
        double value;
        in >> value;

        if (!in.good()) {
            throw std::runtime_error("Deserialization failed: could not read vector value at [" +
                                   std::to_string(i) + "] for '" + expected_label + "'");
        }

        vec[i] = value;
    }

    return vec;
}

std::shared_ptr<ActivationFunction> Serializer::createActivation(const std::string& name) {
    if (name == "sigmoid") {
        return std::make_shared<Sigmoid>();
    } else if (name == "tanh") {
        return std::make_shared<Tanh>();
    } else if (name == "relu") {
        return std::make_shared<ReLU>();
    } else {
        throw std::invalid_argument("Unknown activation function: '" + name +
                                  "'. Valid options are: sigmoid, tanh, relu");
    }
}

void Serializer::expectKeyword(std::istream& in, const std::string& expected) {
    std::string keyword;
    in >> keyword;

    if (!in.good()) {
        throw std::runtime_error("Deserialization failed: could not read keyword (expected '" +
                               expected + "')");
    }

    if (keyword != expected) {
        throw std::runtime_error("Deserialization failed: expected keyword '" + expected +
                               "', found '" + keyword + "'");
    }
}

void Serializer::validateFinite(double value, const std::string& context) {
    if (std::isnan(value)) {
        throw std::invalid_argument("Deserialization failed: NaN value detected in " + context);
    }
    if (std::isinf(value)) {
        throw std::invalid_argument("Deserialization failed: Inf value detected in " + context);
    }
}
