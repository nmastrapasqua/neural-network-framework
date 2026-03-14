#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <cstdint>
#include <iostream>
#include "vector.h"

/**
 * MNISTLoader - Utility class for loading and processing MNIST dataset
 *
 * This header-only utility provides functions to load MNIST data from IDX format files
 * and convert them into formats suitable for neural network training.
 *
 * MNIST IDX Format Specification:
 * - Images file (IDX3): magic number (2051), num_images, rows (28), cols (28), pixel data
 * - Labels file (IDX1): magic number (2049), num_labels, label data
 * - All multi-byte integers are stored in big-endian (MSB first) format
 * - Image pixels are grayscale values in range [0, 255]
 * - Labels are single bytes in range [0, 9]
 */
class MNISTLoader {
public:
    /**
     * Load MNIST images from IDX3 format file
     *
     * @param filename Path to the IDX3 images file
     * @param images Output vector to store loaded images (each image as a Vector of 784 elements)
     * @throws std::runtime_error if file cannot be opened or format is invalid
     */
    static void loadImages(const std::string& filename, std::vector<Vector>& images) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open images file: " + filename);
        }

        // Read and validate header
        uint32_t magic = readInt32(file);
        if (magic != 2051) {
            throw std::runtime_error("Invalid magic number in images file. Expected 2051, got " +
                                   std::to_string(magic));
        }

        uint32_t num_images = readInt32(file);
        uint32_t rows = readInt32(file);
        uint32_t cols = readInt32(file);

        if (rows != 28 || cols != 28) {
            throw std::runtime_error("Invalid image dimensions. Expected 28x28, got " +
                                   std::to_string(rows) + "x" + std::to_string(cols));
        }

        // Read all images
        images.clear();
        images.reserve(num_images);

        const size_t image_size = rows * cols;
        std::vector<uint8_t> image_data(image_size);

        for (uint32_t i = 0; i < num_images; ++i) {
            file.read(reinterpret_cast<char*>(image_data.data()), image_size);

            if (!file) {
                throw std::runtime_error("Failed to read image " + std::to_string(i) +
                                       " from file. File may be corrupted.");
            }

            // Convert to Vector and normalize
            Vector img = imageToVector(image_data);
            normalizeImage(img);
            images.push_back(img);
        }

        file.close();
    }

    /**
     * Load MNIST labels from IDX1 format file
     *
     * @param filename Path to the IDX1 labels file
     * @param labels Output vector to store loaded labels (each label as one-hot encoded Vector)
     * @throws std::runtime_error if file cannot be opened or format is invalid
     */
    static void loadLabels(const std::string& filename, std::vector<Vector>& labels) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open labels file: " + filename);
        }

        // Read and validate header
        uint32_t magic = readInt32(file);
        if (magic != 2049) {
            throw std::runtime_error("Invalid magic number in labels file. Expected 2049, got " +
                                   std::to_string(magic));
        }

        uint32_t num_labels = readInt32(file);

        // Read all labels
        labels.clear();
        labels.reserve(num_labels);

        for (uint32_t i = 0; i < num_labels; ++i) {
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);

            if (!file) {
                throw std::runtime_error("Failed to read label " + std::to_string(i) +
                                       " from file. File may be corrupted.");
            }

            if (label > 9) {
                throw std::runtime_error("Invalid label value: " + std::to_string(label) +
                                       ". Labels must be in range [0, 9].");
            }

            // Convert to one-hot encoding
            labels.push_back(labelToOneHot(label));
        }

        file.close();
    }

    /**
     * Convert raw image data to Vector
     *
     * @param image Raw pixel data (28x28 = 784 bytes)
     * @return Vector of 784 elements containing pixel values
     */
    static Vector imageToVector(const std::vector<uint8_t>& image) {
        Vector vec(image.size());
        for (size_t i = 0; i < image.size(); ++i) {
            vec[i] = static_cast<double>(image[i]);
        }
        return vec;
    }

    /**
     * Convert label to one-hot encoding
     *
     * One-hot encoding represents a label as a vector with all zeros except
     * a single 1 at the index corresponding to the label.
     * Example: label 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
     *
     * @param label Label value (0-9)
     * @param num_classes Number of classes (default: 10 for MNIST)
     * @return Vector with one-hot encoding
     */
    static Vector labelToOneHot(uint8_t label, size_t num_classes = 10) {
        if (label >= num_classes) {
            throw std::invalid_argument("Label " + std::to_string(label) +
                                      " exceeds number of classes " + std::to_string(num_classes));
        }

        Vector one_hot(num_classes, 0.0);
        one_hot[label] = 1.0;
        return one_hot;
    }

    /**
     * Convert one-hot encoding back to label
     *
     * @param one_hot One-hot encoded vector
     * @return Label value (index of maximum element)
     */
    static uint8_t oneHotToLabel(const Vector& one_hot) {
        if (one_hot.size() == 0) {
            throw std::invalid_argument("Cannot convert empty vector to label");
        }

        // Find index of maximum value
        size_t max_index = 0;
        double max_value = one_hot[0];

        for (size_t i = 1; i < one_hot.size(); ++i) {
            if (one_hot[i] > max_value) {
                max_value = one_hot[i];
                max_index = i;
            }
        }

        return static_cast<uint8_t>(max_index);
    }

    /**
     * Normalize image pixels from [0, 255] to [0, 1]
     *
     * Normalization improves neural network training by keeping values in a reasonable range.
     *
     * @param image Image vector to normalize (modified in place)
     */
    static void normalizeImage(Vector& image) {
        for (size_t i = 0; i < image.size(); ++i) {
            image[i] = image[i] / 255.0;
        }
    }

    /**
     * Display an MNIST image as ASCII art
     *
     * This educational function visualizes MNIST images using ASCII characters
     * to represent pixel intensity. Useful for exploring the dataset and
     * understanding what the neural network is learning.
     *
     * @param image Image vector (784 elements, normalized [0,1] or raw [0,255])
     * @param label Label vector (one-hot encoded) or nullptr if not available
     * @param width Image width (default: 28 for MNIST)
     * @param height Image height (default: 28 for MNIST)
     */
    static void displayImage(const Vector& image, const Vector* label = nullptr,
                            size_t width = 28, size_t height = 28) {
        if (image.size() != width * height) {
            throw std::invalid_argument("Image size does not match width * height");
        }

        // ASCII characters representing intensity levels (dark to bright)
        const char* intensity_chars = " .:-=+*#%@";
        const size_t num_chars = 10;

        // Display label if provided
        if (label != nullptr) {
            uint8_t label_value = oneHotToLabel(*label);
            std::cout << "Label: " << static_cast<int>(label_value) << std::endl;
        }

        // Display top border
        std::cout << "+" << std::string(width, '-') << "+" << std::endl;

        // Display image row by row
        for (size_t row = 0; row < height; ++row) {
            std::cout << "|";
            for (size_t col = 0; col < width; ++col) {
                double pixel = image[row * width + col];

                // Normalize to [0, 1] if needed (detect if values are > 1)
                if (pixel > 1.0) {
                    pixel = pixel / 255.0;
                }

                // Map pixel intensity to character index
                size_t char_index = static_cast<size_t>(pixel * (num_chars - 1));
                if (char_index >= num_chars) {
                    char_index = num_chars - 1;
                }

                std::cout << intensity_chars[char_index];
            }
            std::cout << "|" << std::endl;
        }

        // Display bottom border
        std::cout << "+" << std::string(width, '-') << "+" << std::endl;
    }

private:
    /**
     * Read a 32-bit integer in big-endian format
     *
     * The IDX format stores integers in big-endian (most significant byte first),
     * but most modern systems use little-endian. This function handles the conversion.
     *
     * @param file Input file stream
     * @return 32-bit integer value
     */
    static uint32_t readInt32(std::ifstream& file) {
        uint8_t bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);

        if (!file) {
            throw std::runtime_error("Failed to read 32-bit integer from file. Unexpected end of file.");
        }

        // Convert from big-endian to host byte order
        // Big-endian: most significant byte first
        return (static_cast<uint32_t>(bytes[0]) << 24) |
               (static_cast<uint32_t>(bytes[1]) << 16) |
               (static_cast<uint32_t>(bytes[2]) << 8)  |
               (static_cast<uint32_t>(bytes[3]));
    }
};

#endif // MNIST_LOADER_H
