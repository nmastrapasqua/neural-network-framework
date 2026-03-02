#include "matrix.h"
#include "vector.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <stdexcept>

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

void testConstructors() {
    std::cout << "Testing constructors..." << std::endl;

    // Test rows, cols constructor
    Matrix m1(3, 4);
    assert(m1.rows() == 3);
    assert(m1.cols() == 4);

    // Test rows, cols, init_value constructor
    Matrix m2(2, 3, 5.5);
    assert(m2.rows() == 2);
    assert(m2.cols() == 3);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(approxEqual(m2(i, j), 5.5));
        }
    }

    // Test zero rows throws exception
    try {
        Matrix m3(0, 5);
        assert(false && "Should have thrown exception for zero rows");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    // Test zero cols throws exception
    try {
        Matrix m4(5, 0);
        assert(false && "Should have thrown exception for zero cols");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Constructors passed" << std::endl;
}

void testAccessors() {
    std::cout << "Testing accessors..." << std::endl;

    Matrix m(2, 3, 0.0);

    // Test non-const access (modification)
    m(0, 0) = 1.0;
    m(0, 1) = 2.0;
    m(0, 2) = 3.0;
    m(1, 0) = 4.0;
    m(1, 1) = 5.0;
    m(1, 2) = 6.0;

    // Test const access
    assert(approxEqual(m(0, 0), 1.0));
    assert(approxEqual(m(0, 1), 2.0));
    assert(approxEqual(m(0, 2), 3.0));
    assert(approxEqual(m(1, 0), 4.0));
    assert(approxEqual(m(1, 1), 5.0));
    assert(approxEqual(m(1, 2), 6.0));

    // Test dimensions
    assert(m.rows() == 2);
    assert(m.cols() == 3);

    std::cout << "  ✓ Accessors passed" << std::endl;
}

void testTranspose() {
    std::cout << "Testing transpose..." << std::endl;

    Matrix m(2, 3, 0.0);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    Matrix mt = m.transpose();

    // Check dimensions
    assert(mt.rows() == 3);
    assert(mt.cols() == 2);

    // Check values
    assert(approxEqual(mt(0, 0), 1.0));
    assert(approxEqual(mt(0, 1), 4.0));
    assert(approxEqual(mt(1, 0), 2.0));
    assert(approxEqual(mt(1, 1), 5.0));
    assert(approxEqual(mt(2, 0), 3.0));
    assert(approxEqual(mt(2, 1), 6.0));

    // Test transpose involution: (M^T)^T = M
    Matrix mtt = mt.transpose();
    assert(mtt.rows() == m.rows());
    assert(mtt.cols() == m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            assert(approxEqual(mtt(i, j), m(i, j)));
        }
    }

    std::cout << "  ✓ Transpose passed" << std::endl;
}

void testAddition() {
    std::cout << "Testing addition..." << std::endl;

    Matrix m1(2, 2, 0.0);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0;
    m1(1, 0) = 3.0; m1(1, 1) = 4.0;

    Matrix m2(2, 2, 0.0);
    m2(0, 0) = 5.0; m2(0, 1) = 6.0;
    m2(1, 0) = 7.0; m2(1, 1) = 8.0;

    Matrix result = m1 + m2;

    assert(result.rows() == 2);
    assert(result.cols() == 2);
    assert(approxEqual(result(0, 0), 6.0));
    assert(approxEqual(result(0, 1), 8.0));
    assert(approxEqual(result(1, 0), 10.0));
    assert(approxEqual(result(1, 1), 12.0));

    // Test dimension mismatch
    Matrix m3(2, 3, 0.0);
    try {
        Matrix bad = m1 + m3;
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Addition passed" << std::endl;
}

void testMatrixMultiplication() {
    std::cout << "Testing matrix multiplication..." << std::endl;

    // Test 2x3 * 3x2 = 2x2
    Matrix m1(2, 3, 0.0);
    m1(0, 0) = 1.0; m1(0, 1) = 2.0; m1(0, 2) = 3.0;
    m1(1, 0) = 4.0; m1(1, 1) = 5.0; m1(1, 2) = 6.0;

    Matrix m2(3, 2, 0.0);
    m2(0, 0) = 7.0;  m2(0, 1) = 8.0;
    m2(1, 0) = 9.0;  m2(1, 1) = 10.0;
    m2(2, 0) = 11.0; m2(2, 1) = 12.0;

    Matrix result = m1 * m2;

    assert(result.rows() == 2);
    assert(result.cols() == 2);

    // result(0,0) = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    assert(approxEqual(result(0, 0), 58.0));
    // result(0,1) = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    assert(approxEqual(result(0, 1), 64.0));
    // result(1,0) = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    assert(approxEqual(result(1, 0), 139.0));
    // result(1,1) = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
    assert(approxEqual(result(1, 1), 154.0));

    // Test identity matrix property
    Matrix identity(2, 2, 0.0);
    identity(0, 0) = 1.0;
    identity(1, 1) = 1.0;

    Matrix m3(2, 2, 0.0);
    m3(0, 0) = 3.0; m3(0, 1) = 4.0;
    m3(1, 0) = 5.0; m3(1, 1) = 6.0;

    Matrix result2 = m3 * identity;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            assert(approxEqual(result2(i, j), m3(i, j)));
        }
    }

    // Test dimension mismatch
    Matrix m4(2, 2, 0.0);
    Matrix m5(3, 2, 0.0);
    try {
        Matrix bad = m4 * m5;
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Matrix multiplication passed" << std::endl;
}

void testMatrixVectorMultiplication() {
    std::cout << "Testing matrix-vector multiplication..." << std::endl;

    Matrix m(2, 3, 0.0);
    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;

    Vector v{7.0, 8.0, 9.0};

    Vector result = m * v;

    assert(result.size() == 2);
    // result[0] = 1*7 + 2*8 + 3*9 = 7 + 16 + 27 = 50
    assert(approxEqual(result[0], 50.0));
    // result[1] = 4*7 + 5*8 + 6*9 = 28 + 40 + 54 = 122
    assert(approxEqual(result[1], 122.0));

    // Test dimension mismatch
    Vector v2{1.0, 2.0};
    try {
        Vector bad = m * v2;
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Matrix-vector multiplication passed" << std::endl;
}

void testElementWiseMultiply() {
    std::cout << "Testing element-wise multiplication..." << std::endl;

    Matrix m1(2, 2, 0.0);
    m1(0, 0) = 2.0; m1(0, 1) = 3.0;
    m1(1, 0) = 4.0; m1(1, 1) = 5.0;

    Matrix m2(2, 2, 0.0);
    m2(0, 0) = 6.0; m2(0, 1) = 7.0;
    m2(1, 0) = 8.0; m2(1, 1) = 9.0;

    Matrix result = m1.elementWiseMultiply(m2);

    assert(result.rows() == 2);
    assert(result.cols() == 2);
    assert(approxEqual(result(0, 0), 12.0));
    assert(approxEqual(result(0, 1), 21.0));
    assert(approxEqual(result(1, 0), 32.0));
    assert(approxEqual(result(1, 1), 45.0));

    // Test dimension mismatch
    Matrix m3(2, 3, 0.0);
    try {
        Matrix bad = m1.elementWiseMultiply(m3);
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Element-wise multiplication passed" << std::endl;
}

void testUtilities() {
    std::cout << "Testing utilities..." << std::endl;

    // Test fill
    Matrix m1(3, 2);
    m1.fill(7.5);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            assert(approxEqual(m1(i, j), 7.5));
        }
    }

    // Test randomize
    Matrix m2(3, 3);
    m2.randomize(-1.0, 1.0);

    // Check that values are within range
    bool hasVariation = false;
    double firstValue = m2(0, 0);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            assert(m2(i, j) >= -1.0 && m2(i, j) <= 1.0);
            if (!approxEqual(m2(i, j), firstValue)) {
                hasVariation = true;
            }
        }
    }
    // With high probability, not all values should be identical
    assert(hasVariation && "Randomize should produce varied values");

    std::cout << "  ✓ Utilities passed" << std::endl;
}

int main() {
    std::cout << "Running Matrix tests..." << std::endl;
    std::cout << "======================" << std::endl;

    testConstructors();
    testAccessors();
    testTranspose();
    testAddition();
    testMatrixMultiplication();
    testMatrixVectorMultiplication();
    testElementWiseMultiply();
    testUtilities();

    std::cout << std::endl;
    std::cout << "All Matrix tests passed! ✓" << std::endl;

    return 0;
}
