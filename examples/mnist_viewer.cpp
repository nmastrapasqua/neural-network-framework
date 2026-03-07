#include "mnist_loader.h"
#include <iostream>
#include <limits>
#include <string>

/**
 * MNIST Viewer - Interactive Dataset Explorer
 *
 * This interactive program allows you to explore the MNIST dataset by
 * viewing individual images as ASCII art along with their labels.
 *
 * Features:
 * - Load MNIST training or test dataset
 * - Browse images by index
 * - View ASCII art representation of digits
 * - See the corresponding label
 * - Educational tool for understanding the dataset
 */

void clearScreen() {
    // Simple cross-platform screen clear
    #ifdef _WIN32
        system("cls");
    #else
        system("clear");
    #endif
}

void displayMenu(size_t total_images) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "  MNIST Dataset Viewer" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total images loaded: " << total_images << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  - Enter an index (0-" << (total_images - 1) << ") to view an image" << std::endl;
    std::cout << "  - Enter 'q' or 'quit' to exit" << std::endl;
    std::cout << "  - Enter 'r' or 'random' to view a random image" << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // Determine which dataset to load (default: training set)
        std::string images_file = "mnist-dataset/train-images.idx3-ubyte";
        std::string labels_file = "mnist-dataset/train-labels.idx1-ubyte";
        std::string dataset_name = "Training";

        // Check command line arguments
        if (argc > 1) {
            std::string arg = argv[1];
            if (arg == "test" || arg == "--test" || arg == "-t") {
                images_file = "mnist-dataset/t10k-images.idx3-ubyte";
                labels_file = "mnist-dataset/t10k-labels.idx1-ubyte";
                dataset_name = "Test";
            }
        }

        std::cout << "========================================" << std::endl;
        std::cout << "  MNIST Dataset Viewer" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "\nLoading " << dataset_name << " dataset..." << std::endl;

        // Load images and labels
        std::vector<Vector> images;
        std::vector<Vector> labels;

        std::cout << "Loading images from: " << images_file << std::endl;
        MNISTLoader::loadImages(images_file, images);

        std::cout << "Loading labels from: " << labels_file << std::endl;
        MNISTLoader::loadLabels(labels_file, labels);

        std::cout << "\n✓ Successfully loaded " << images.size() << " images!" << std::endl;
        std::cout << "\nPress Enter to continue...";
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

        // Interactive loop
        while (true) {
            clearScreen();
            displayMenu(images.size());

            std::cout << "\nYour choice: ";
            std::string input;
            std::getline(std::cin, input);

            // Check for quit command
            if (input == "q" || input == "quit" || input == "exit") {
                std::cout << "\nThank you for using MNIST Viewer!" << std::endl;
                break;
            }

            // Check for random command
            if (input == "r" || input == "random") {
                size_t random_index = rand() % images.size();
                std::cout << "\nShowing random image at index: " << random_index << std::endl;
                std::cout << std::endl;
                MNISTLoader::displayImage(images[random_index], &labels[random_index]);

                std::cout << "\nPress Enter to continue...";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                continue;
            }

            // Try to parse as index
            try {
                size_t index = std::stoull(input);

                if (index >= images.size()) {
                    std::cout << "\n✗ Error: Index out of range!" << std::endl;
                    std::cout << "  Valid range: 0-" << (images.size() - 1) << std::endl;
                    std::cout << "\nPress Enter to continue...";
                    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                    continue;
                }

                // Display the selected image
                std::cout << "\nShowing image at index: " << index << std::endl;
                std::cout << std::endl;
                MNISTLoader::displayImage(images[index], &labels[index]);

                std::cout << "\nPress Enter to continue...";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

            } catch (const std::exception& e) {
                std::cout << "\n✗ Invalid input! Please enter a number, 'r' for random, or 'q' to quit." << std::endl;
                std::cout << "\nPress Enter to continue...";
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
        }

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        std::cerr << "\nUsage: " << std::endl;
        std::cerr << "  " << (argc > 0 ? argv[0] : "./mnist_viewer") << "           # View training set (default)" << std::endl;
        std::cerr << "  " << (argc > 0 ? argv[0] : "./mnist_viewer") << " test      # View test set" << std::endl;
        return 1;
    }
}
