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

    // Test size constructor
    Vector v1(5);
    assert(v1.size() == 5);

    // Test size + init_value constructor
    Vector v2(3, 2.5);
    assert(v2.size() == 3);
    assert(approxEqual(v2[0], 2.5));
    assert(approxEqual(v2[1], 2.5));
    assert(approxEqual(v2[2], 2.5));

    // Test initializer_list constructor
    Vector v3{1.0, 2.0, 3.0, 4.0};
    assert(v3.size() == 4);
    assert(approxEqual(v3[0], 1.0));
    assert(approxEqual(v3[1], 2.0));
    assert(approxEqual(v3[2], 3.0));
    assert(approxEqual(v3[3], 4.0));

    // Test zero size throws exception
    try {
        Vector v4(0);
        assert(false && "Should have thrown exception for zero size");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Constructors passed" << std::endl;
}

void testAccessors() {
    std::cout << "Testing accessors..." << std::endl;

    Vector v{1.0, 2.0, 3.0};

    // Test const access
    assert(approxEqual(v[0], 1.0));
    assert(approxEqual(v[1], 2.0));
    assert(approxEqual(v[2], 3.0));

    // Test non-const access (modification)
    v[1] = 5.0;
    assert(approxEqual(v[1], 5.0));

    // Test size
    assert(v.size() == 3);

    std::cout << "  ✓ Accessors passed" << std::endl;
}

void testAddition() {
    std::cout << "Testing addition..." << std::endl;

    Vector v1{1.0, 2.0, 3.0};
    Vector v2{4.0, 5.0, 6.0};

    Vector result = v1 + v2;
    assert(result.size() == 3);
    assert(approxEqual(result[0], 5.0));
    assert(approxEqual(result[1], 7.0));
    assert(approxEqual(result[2], 9.0));

    // Test dimension mismatch
    Vector v3{1.0, 2.0};
    try {
        Vector bad = v1 + v3;
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Addition passed" << std::endl;
}

void testSubtraction() {
    std::cout << "Testing subtraction..." << std::endl;

    Vector v1{5.0, 7.0, 9.0};
    Vector v2{1.0, 2.0, 3.0};

    Vector result = v1 - v2;
    assert(result.size() == 3);
    assert(approxEqual(result[0], 4.0));
    assert(approxEqual(result[1], 5.0));
    assert(approxEqual(result[2], 6.0));

    // Test dimension mismatch
    Vector v3{1.0, 2.0};
    try {
        Vector bad = v1 - v3;
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Subtraction passed" << std::endl;
}

void testElementWiseMultiply() {
    std::cout << "Testing element-wise multiplication..." << std::endl;

    Vector v1{2.0, 3.0, 4.0};
    Vector v2{5.0, 6.0, 7.0};

    Vector result = v1.elementWiseMultiply(v2);
    assert(result.size() == 3);
    assert(approxEqual(result[0], 10.0));
    assert(approxEqual(result[1], 18.0));
    assert(approxEqual(result[2], 28.0));

    // Test dimension mismatch
    Vector v3{1.0, 2.0};
    try {
        Vector bad = v1.elementWiseMultiply(v3);
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Element-wise multiplication passed" << std::endl;
}

void testDotProduct() {
    std::cout << "Testing dot product..." << std::endl;

    Vector v1{1.0, 2.0, 3.0};
    Vector v2{4.0, 5.0, 6.0};

    double result = v1.dot(v2);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert(approxEqual(result, 32.0));

    // Test with orthogonal vectors
    Vector v3{1.0, 0.0};
    Vector v4{0.0, 1.0};
    assert(approxEqual(v3.dot(v4), 0.0));

    // Test dimension mismatch
    Vector v5{1.0, 2.0};
    try {
        double bad = v1.dot(v5);
        (void)bad; // Suppress unused variable warning
        assert(false && "Should have thrown exception for dimension mismatch");
    } catch (const std::invalid_argument& e) {
        // Expected
    }

    std::cout << "  ✓ Dot product passed" << std::endl;
}

void testUtilities() {
    std::cout << "Testing utilities..." << std::endl;

    // Test fill
    Vector v1(4);
    v1.fill(7.5);
    assert(approxEqual(v1[0], 7.5));
    assert(approxEqual(v1[1], 7.5));
    assert(approxEqual(v1[2], 7.5));
    assert(approxEqual(v1[3], 7.5));

    // Test sum
    Vector v2{1.0, 2.0, 3.0, 4.0};
    assert(approxEqual(v2.sum(), 10.0));

    // Test mean
    assert(approxEqual(v2.mean(), 2.5));

    Vector v3{5.0, 10.0, 15.0};
    assert(approxEqual(v3.mean(), 10.0));

    std::cout << "  ✓ Utilities passed" << std::endl;
}

int main() {
    std::cout << "Running Vector tests..." << std::endl;
    std::cout << "======================" << std::endl;

    testConstructors();
    testAccessors();
    testAddition();
    testSubtraction();
    testElementWiseMultiply();
    testDotProduct();
    testUtilities();

    std::cout << std::endl;
    std::cout << "All Vector tests passed! ✓" << std::endl;

    return 0;
}
