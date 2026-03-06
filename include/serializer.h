#ifndef SERIALIZER_H
#define SERIALIZER_H

#include <ostream>
#include <istream>
#include <string>
#include <memory>
#include "matrix.h"
#include "vector.h"
#include "activation.h"

// Forward declaration
class Network;

/**
 * Serializer class provides functionality to save and load neural networks.
 *
 * The serialization format is a human-readable text format:
 * - Header: "NEURAL_NETWORK_V1"
 * - Number of layers
 * - For each layer:
 *   - Input size, output size
 *   - Activation function name
 *   - Weights matrix
 *   - Biases vector
 * - Footer: "END"
 *
 * Requirements validated:
 * - 10.1: Serialize network to text format
 * - 10.2: Save all weights, biases, and network configuration
 * - 10.3: Save activation function type for each layer
 * - 10.8: Implement pretty printer for serialization format
 */
class Serializer {
public:
    /**
     * Serialize a network to an output stream.
     *
     * Writes the complete network structure including:
     * - Network topology (layer sizes)
     * - Activation functions for each layer
     * - All weights and biases
     *
     * Format:
     * NEURAL_NETWORK_V1
     * LAYERS <num_layers>
     * LAYER 0
     *   INPUT_SIZE <size>
     *   OUTPUT_SIZE <size>
     *   ACTIVATION <name>
     *   WEIGHTS <rows> <cols>
     *     <weight values...>
     *   BIASES <size>
     *     <bias values...>
     * ...
     * END
     *
     * @param network Network to serialize
     * @param out Output stream to write to
     * @throws std::runtime_error if writing fails
     */
    static void serialize(const Network& network, std::ostream& out);

    /**
     * Deserialize a network from an input stream.
     *
     * Reads the serialization format and reconstructs a Network object.
     *
     * Validates:
     * - File format header (NEURAL_NETWORK_V1)
     * - All required keywords (LAYERS, LAYER, INPUT_SIZE, etc.)
     * - Dimension consistency (layer connectivity)
     * - Numerical values (no NaN, Inf)
     * - File completeness (END marker)
     *
     * Requirements validated:
     * - 10.4: Reconstruct network with same architecture and parameters
     * - 10.5: Validate file integrity during loading
     * - 10.6: Signal descriptive error if file is corrupted/incompatible
     * - 10.7: Implement parser for serialization format
     *
     * @param in Input stream to read from
     * @return Reconstructed Network object
     * @throws std::runtime_error if file format is invalid
     * @throws std::invalid_argument if data is corrupted or incompatible
     */
    static Network deserialize(std::istream& in);

private:
    /**
     * Write a matrix to an output stream.
     *
     * Format:
     * WEIGHTS <rows> <cols>
     *   <row 0 values...>
     *   <row 1 values...>
     *   ...
     *
     * @param mat Matrix to write
     * @param out Output stream
     * @param label Label for the matrix (e.g., "WEIGHTS")
     */
    static void writeMatrix(const Matrix& mat, std::ostream& out, const std::string& label);

    /**
     * Write a vector to an output stream.
     *
     * Format:
     * BIASES <size>
     *   <value 0> <value 1> ...
     *
     * @param vec Vector to write
     * @param out Output stream
     * @param label Label for the vector (e.g., "BIASES")
     */
    static void writeVector(const Vector& vec, std::ostream& out, const std::string& label);

    /**
     * Read a matrix from an input stream.
     *
     * Expected format:
     * WEIGHTS <rows> <cols>
     *   <row 0 values...>
     *   <row 1 values...>
     *   ...
     *
     * @param in Input stream
     * @param expected_label Expected label (e.g., "WEIGHTS")
     * @return Matrix read from stream
     * @throws std::runtime_error if format is invalid
     */
    static Matrix readMatrix(std::istream& in, const std::string& expected_label);

    /**
     * Read a vector from an input stream.
     *
     * Expected format:
     * BIASES <size>
     *   <value 0> <value 1> ...
     *
     * @param in Input stream
     * @param expected_label Expected label (e.g., "BIASES")
     * @return Vector read from stream
     * @throws std::runtime_error if format is invalid
     */
    static Vector readVector(std::istream& in, const std::string& expected_label);

    /**
     * Create an activation function from its name.
     *
     * @param name Activation function name ("sigmoid", "tanh", "relu")
     * @return Shared pointer to activation function
     * @throws std::invalid_argument if name is not recognized
     */
    static std::shared_ptr<ActivationFunction> createActivation(const std::string& name);

    /**
     * Read and validate a keyword from the input stream.
     *
     * @param in Input stream
     * @param expected Expected keyword
     * @throws std::runtime_error if keyword doesn't match
     */
    static void expectKeyword(std::istream& in, const std::string& expected);

    /**
     * Validate that a numeric value is finite (not NaN or Inf).
     *
     * @param value Value to check
     * @param context Context string for error message
     * @throws std::invalid_argument if value is not finite
     */
    static void validateFinite(double value, const std::string& context);
};

#endif // SERIALIZER_H
