#include "vector.h"
#include "matrix.h"
#include <iostream>

/**
 * Demo: Print Methods for Vector and Matrix
 *
 * This example demonstrates the print() methods for educational purposes.
 * Useful for debugging and understanding what's happening inside the network.
 */

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "  Vector and Matrix Print Demo" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // ============================================================================
    // Vector Examples
    // ============================================================================

    std::cout << "VECTOR EXAMPLES:" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << std::endl;

    // Example 1: Simple vector
    Vector v1{1.0, 2.0, 3.0, 4.0, 5.0};
    v1.print("v1");
    std::cout << std::endl;

    // Example 2: Vector with name
    Vector v2(10, 0.5);
    v2.print("v2 (initialized with 0.5)");
    std::cout << std::endl;

    // Example 3: Vector without name
    Vector v3{-1.5, 2.7, -3.2, 4.8, 1.2};  // Same size as v1
    std::cout << "Anonymous vector: ";
    v3.print();
    std::cout << std::endl;

    // Example 4: Vector operations result
    Vector v4 = v1 + v3;
    v4.print("v1 + v3");
    std::cout << std::endl;

    // Example 5: Large vector (shows formatting)
    Vector v5(20);
    for (size_t i = 0; i < v5.size(); ++i) {
        v5[i] = static_cast<double>(i) * 0.1;
    }
    v5.print("Large vector (20 elements)");
    std::cout << std::endl;

    // ============================================================================
    // Matrix Examples
    // ============================================================================

    std::cout << "MATRIX EXAMPLES:" << std::endl;
    std::cout << "----------------" << std::endl;
    std::cout << std::endl;

    // Example 1: Small matrix
    Matrix m1(3, 3);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            m1(i, j) = static_cast<double>(i * 3 + j + 1);
        }
    }
    m1.print("m1 (3x3 matrix)");
    std::cout << std::endl;

    // Example 2: Identity-like matrix
    Matrix m2(4, 4, 0.0);
    for (size_t i = 0; i < 4; ++i) {
        m2(i, i) = 1.0;
    }
    m2.print("m2 (4x4 identity)");
    std::cout << std::endl;

    // Example 3: Rectangular matrix
    Matrix m3(2, 5);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 5; ++j) {
            m3(i, j) = (i + 1) * (j + 1) * 0.5;
        }
    }
    m3.print("m3 (2x5 rectangular)");
    std::cout << std::endl;

    // Example 4: Matrix with negative values
    Matrix m4(3, 3);
    m4(0, 0) = -1.5; m4(0, 1) = 2.3;  m4(0, 2) = -0.7;
    m4(1, 0) = 4.2;  m4(1, 1) = -3.1; m4(1, 2) = 1.9;
    m4(2, 0) = -2.8; m4(2, 1) = 0.6;  m4(2, 2) = -4.5;
    m4.print("m4 (mixed positive/negative)");
    std::cout << std::endl;

    // Example 5: Matrix without name
    Matrix m5(2, 3, 7.5);
    std::cout << "Anonymous matrix:" << std::endl;
    m5.print();
    std::cout << std::endl;

    // Example 6: Transpose demonstration
    Matrix m6(2, 4);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            m6(i, j) = i * 10 + j;
        }
    }
    m6.print("m6 (original 2x4)");
    std::cout << std::endl;

    Matrix m6_t = m6.transpose();
    m6_t.print("m6 transposed (4x2)");
    std::cout << std::endl;

    // Example 7: Matrix-vector multiplication
    Matrix m7(3, 4);
    m7.randomize(-1.0, 1.0);
    m7.print("m7 (3x4 random weights)");
    std::cout << std::endl;

    Vector v7{1.0, 2.0, 3.0, 4.0};
    v7.print("v7 (input vector)");
    std::cout << std::endl;

    Vector result = m7 * v7;
    result.print("m7 * v7 (result)");
    std::cout << std::endl;

    // ============================================================================
    // Educational Use Cases
    // ============================================================================

    std::cout << "EDUCATIONAL USE CASES:" << std::endl;
    std::cout << "----------------------" << std::endl;
    std::cout << std::endl;

    std::cout << "Use print() methods to:" << std::endl;
    std::cout << "1. Debug network weights and biases" << std::endl;
    std::cout << "2. Visualize forward pass activations" << std::endl;
    std::cout << "3. Inspect gradient values during backpropagation" << std::endl;
    std::cout << "4. Understand matrix operations step-by-step" << std::endl;
    std::cout << "5. Verify initialization methods (Xavier, He)" << std::endl;
    std::cout << std::endl;

    std::cout << "Example: Inspecting layer weights" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    Matrix weights(2, 3);
    weights.randomize(-0.5, 0.5);
    weights.print("Layer weights");

    Vector biases(2, 0.0);
    biases.print("Layer biases");
    std::cout << std::endl;

    std::cout << "Demo completed successfully!" << std::endl;

    return 0;
}
