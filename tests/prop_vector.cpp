#include "vector.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <iostream>
#include <cmath>
#include <limits>
#include <random>

// Helper function to compare doubles with tolerance
bool approxEqual(double a, double b, double epsilon = 1e-9) {
    return std::abs(a - b) < epsilon;
}

// Custom generator for doubles in a range
rc::Gen<double> genDoubleInRange(double min, double max) {
    return rc::gen::map(rc::gen::inRange(0, 10000), [min, max](int i) {
        return min + (max - min) * (i / 10000.0);
    });
}

// Generator for vectors of specific size
rc::Gen<Vector> arbVectorOfSize(size_t size) {
    return rc::gen::apply([size](const std::vector<double>& values) {
        Vector v(size);
        for (size_t i = 0; i < size && i < values.size(); ++i) {
            v[i] = values[i];
        }
        return v;
    }, rc::gen::container<std::vector<double>>(size, genDoubleInRange(-100.0, 100.0)));
}

// Custom generator for Vector
namespace rc {
    template<>
    struct Arbitrary<Vector> {
        static Gen<Vector> arbitrary() {
            return gen::apply([](size_t size) {
                if (size == 0) {
                    size = 1;  // Ensure non-empty vectors
                }
                return *arbVectorOfSize(size);
            }, gen::inRange<size_t>(1, 20));
        }
    };
}

// **Validates: Requirements 9.4**
// Feature: neural-network-framework, Property 25: Vector Operations Correctness
// Verify that vector operations (addition, subtraction, element-wise multiplication, dot product)
// match their mathematical definitions
RC_GTEST_PROP(VectorPropertyTest, VectorOperationsCorrectness, ()) {
    // Generate a random size for vectors
    auto size = *rc::gen::inRange<size_t>(1, 20);

    // Generate two vectors of the same size
    auto u = *arbVectorOfSize(size);
    auto v = *arbVectorOfSize(size);

    // Test addition: (u + v)[i] = u[i] + v[i]
    Vector sum = u + v;
    RC_ASSERT(sum.size() == size);
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(sum[i], u[i] + v[i]));
    }

    // Test subtraction: (u - v)[i] = u[i] - v[i]
    Vector diff = u - v;
    RC_ASSERT(diff.size() == size);
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(diff[i], u[i] - v[i]));
    }

    // Test element-wise multiplication: (u * v)[i] = u[i] * v[i]
    Vector prod = u.elementWiseMultiply(v);
    RC_ASSERT(prod.size() == size);
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(prod[i], u[i] * v[i]));
    }

    // Test dot product: u · v = Σ(u[i] * v[i])
    double dotProduct = u.dot(v);
    double expectedDot = 0.0;
    for (size_t i = 0; i < size; ++i) {
        expectedDot += u[i] * v[i];
    }
    RC_ASSERT(approxEqual(dotProduct, expectedDot));
}

// **Validates: Requirements 9.4**
// Feature: neural-network-framework, Property 35: Vector Addition Commutativity
// Verify that for any vectors u and v of the same size, u + v equals v + u
RC_GTEST_PROP(VectorPropertyTest, VectorAdditionCommutativity, ()) {
    // Generate a random size for vectors
    auto size = *rc::gen::inRange<size_t>(1, 20);

    // Generate two vectors of the same size
    auto u = *arbVectorOfSize(size);
    auto v = *arbVectorOfSize(size);

    // Compute u + v and v + u
    Vector uPlusV = u + v;
    Vector vPlusU = v + u;

    // Verify they are equal
    RC_ASSERT(uPlusV.size() == vPlusU.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(uPlusV[i], vPlusU[i]));
    }
}

// Additional property: Vector addition associativity
// (u + v) + w = u + (v + w)
RC_GTEST_PROP(VectorPropertyTest, VectorAdditionAssociativity, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);

    auto u = *arbVectorOfSize(size);
    auto v = *arbVectorOfSize(size);
    auto w = *arbVectorOfSize(size);

    Vector left = (u + v) + w;
    Vector right = u + (v + w);

    RC_ASSERT(left.size() == right.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(left[i], right[i]));
    }
}

// Additional property: Vector addition identity
// u + 0 = u
RC_GTEST_PROP(VectorPropertyTest, VectorAdditionIdentity, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto u = *arbVectorOfSize(size);

    Vector zero(size, 0.0);
    Vector result = u + zero;

    RC_ASSERT(result.size() == u.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(result[i], u[i]));
    }
}

// Additional property: Vector subtraction inverse
// u - u = 0
RC_GTEST_PROP(VectorPropertyTest, VectorSubtractionInverse, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto u = *arbVectorOfSize(size);

    Vector result = u - u;

    RC_ASSERT(result.size() == u.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(result[i], 0.0));
    }
}

// Additional property: Dot product commutativity
// u · v = v · u
RC_GTEST_PROP(VectorPropertyTest, DotProductCommutativity, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);

    auto u = *arbVectorOfSize(size);
    auto v = *arbVectorOfSize(size);

    double uDotV = u.dot(v);
    double vDotU = v.dot(u);

    RC_ASSERT(approxEqual(uDotV, vDotU));
}

// Additional property: Dot product with zero vector
// u · 0 = 0
RC_GTEST_PROP(VectorPropertyTest, DotProductWithZero, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto u = *arbVectorOfSize(size);

    Vector zero(size, 0.0);
    double result = u.dot(zero);

    RC_ASSERT(approxEqual(result, 0.0));
}

// Additional property: Dot product with self is non-negative
// u · u >= 0
RC_GTEST_PROP(VectorPropertyTest, DotProductWithSelfNonNegative, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto u = *arbVectorOfSize(size);

    double result = u.dot(u);

    RC_ASSERT(result >= -1e-9);  // Allow small negative due to floating point errors
}

// Additional property: Element-wise multiplication commutativity
// u ⊙ v = v ⊙ u
RC_GTEST_PROP(VectorPropertyTest, ElementWiseMultiplicationCommutativity, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);

    auto u = *arbVectorOfSize(size);
    auto v = *arbVectorOfSize(size);

    Vector uTimesV = u.elementWiseMultiply(v);
    Vector vTimesU = v.elementWiseMultiply(u);

    RC_ASSERT(uTimesV.size() == vTimesU.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(uTimesV[i], vTimesU[i]));
    }
}

// Additional property: Element-wise multiplication with ones is identity
// u ⊙ 1 = u
RC_GTEST_PROP(VectorPropertyTest, ElementWiseMultiplicationIdentity, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto u = *arbVectorOfSize(size);

    Vector ones(size, 1.0);
    Vector result = u.elementWiseMultiply(ones);

    RC_ASSERT(result.size() == u.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(result[i], u[i]));
    }
}

// Additional property: Element-wise multiplication with zero
// u ⊙ 0 = 0
RC_GTEST_PROP(VectorPropertyTest, ElementWiseMultiplicationWithZero, ()) {
    auto size = *rc::gen::inRange<size_t>(1, 20);
    auto u = *arbVectorOfSize(size);

    Vector zero(size, 0.0);
    Vector result = u.elementWiseMultiply(zero);

    RC_ASSERT(result.size() == u.size());
    for (size_t i = 0; i < size; ++i) {
        RC_ASSERT(approxEqual(result[i], 0.0));
    }
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
