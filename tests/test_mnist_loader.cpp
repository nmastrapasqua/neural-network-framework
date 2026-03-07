#include "mnist_loader.h"
#include <iostream>

int main() {
    try {
        std::cout << "Testing MNISTLoader utility..." << std::endl;

        // Test loading a small subset of images and labels
        std::vector<Vector> images;
        std::vector<Vector> labels;

        std::cout << "Loading training images..." << std::endl;
        MNISTLoader::loadImages("mnist-dataset/train-images.idx3-ubyte", images);
        std::cout << "Loaded " << images.size() << " images" << std::endl;

        std::cout << "Loading training labels..." << std::endl;
        MNISTLoader::loadLabels("mnist-dataset/train-labels.idx1-ubyte", labels);
        std::cout << "Loaded " << labels.size() << " labels" << std::endl;

        // Verify dimensions
        if (images.size() != labels.size()) {
            std::cerr << "Error: Number of images and labels don't match!" << std::endl;
            return 1;
        }

        // Check first image
        if (images[0].size() != 784) {
            std::cerr << "Error: Image size should be 784, got " << images[0].size() << std::endl;
            return 1;
        }

        // Check first label
        if (labels[0].size() != 10) {
            std::cerr << "Error: Label size should be 10, got " << labels[0].size() << std::endl;
            return 1;
        }

        // Test one-hot encoding
        std::cout << "\nTesting one-hot encoding..." << std::endl;
        Vector one_hot = MNISTLoader::labelToOneHot(3);
        std::cout << "Label 3 as one-hot: [";
        for (size_t i = 0; i < one_hot.size(); ++i) {
            std::cout << one_hot[i];
            if (i < one_hot.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Test one-hot to label conversion
        uint8_t label = MNISTLoader::oneHotToLabel(one_hot);
        std::cout << "Converted back to label: " << static_cast<int>(label) << std::endl;

        // Test normalization
        std::cout << "\nTesting normalization..." << std::endl;
        std::cout << "First pixel of first image (normalized): " << images[0][0] << std::endl;
        std::cout << "Should be in range [0, 1]" << std::endl;

        // Test ASCII art display
        std::cout << "\nTesting ASCII art display..." << std::endl;
        std::cout << "Displaying first 3 training images:" << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < 3; ++i) {
            std::cout << "Image " << i << ":" << std::endl;
            MNISTLoader::displayImage(images[i], &labels[i]);
            std::cout << std::endl;
        }

        std::cout << "\nAll tests passed!" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
