#include "matrix.h"
#include "vector.h"
#include <gtest/gtest.h>
#include <rapidcheck.h>
#include <rapidcheck/gtest.h>
#include <iostream>
#include <cmath>
#include <limits>

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

// Generator for matrices of specific size
rc::Gen<Matrix> arbMatrixOfSize(size_t rows, size_t cols) {
    return rc::gen::apply([rows, cols](const std::vector<double>& values) {
        Matrix m(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                size_t idx = i * cols + j;
                if (idx < values.size()) {
                    m(i, j) = values[idx];
                }
            }
        }
        return m;
    }, rc::gen::container<std::vector<double>>(rows * cols, genDoubleInRange(-10.0, 10.0)));
}

// Generator for vectors of specific size
rc::Gen<Vector> arbVectorOfSize(size_t size) {
    return rc::gen::apply([size](const std::vector<double>& values) {
        Vector v(size);
        for (size_t i = 0; i < size && i < values.size(); ++i) {
            v[i] = values[i];
        }
        return v;
    }, rc::gen::container<std::vector<double>>(size, genDoubleInRange(-10.0, 10.0)));
}

// **Validates: Requirements 9.3**
// Feature: neural-network-framework, Property 24: Matrix Operations Correctness
// Verify that matrix operations (multiplication, transpose, addition, element-wise operations)
// match their mathematical definitions
RC_GTEST_PROP(MatrixPropertyTest, MatrixOperationsCorrectness, ()) {
    // Generate random dimensions
    auto m = *rc::gen::inRange<size_t>(1, 10);
    auto n = *rc::gen::inRange<size_t>(1, 10);
    auto p = *rc::gen::inRange<size_t>(1, 10);

    // Generate matrices and vector
    auto A = *arbMatrixOfSize(m, n);
    auto B = *arbMatrixOfSize(n, p);
    auto C = *arbMatrixOfSize(m, n);  // Same size as A for addition
    auto v = *arbVectorOfSize(n);

    // Test 1: Matrix-Matrix Multiplication
    // C[i][j] = Σ(A[i][k] * B[k][j]) for k=0 to n-1
    Matrix AB = A * B;
    RC_ASSERT(AB.rows() == m);
    RC_ASSERT(AB.cols() == p);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < n; ++k) {
                expected += A(i, k) * B(k, j);
            }
            RC_ASSERT(approxEqual(AB(i, j), expected, 1e-6));
        }
    }

    // Test 2: Matrix-Vector Multiplication
    // result[i] = Σ(A[i][j] * v[j]) for j=0 to n-1
    Vector Av = A * v;
    RC_ASSERT(Av.size() == m);

    for (size_t i = 0; i < m; ++i) {
        double expected = 0.0;
        for (size_t j = 0; j < n; ++j) {
            expected += A(i, j) * v[j];
        }
        RC_ASSERT(approxEqual(Av[i], expected, 1e-6));
    }

    // Test 3: Matrix Transpose
    // A^T[j][i] = A[i][j]
    Matrix AT = A.transpose();
    RC_ASSERT(AT.rows() == n);
    RC_ASSERT(AT.cols() == m);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            RC_ASSERT(approxEqual(AT(j, i), A(i, j)));
        }
    }

    // Test 4: Matrix Addition
    // (A + C)[i][j] = A[i][j] + C[i][j]
    Matrix AC = A + C;
    RC_ASSERT(AC.rows() == m);
    RC_ASSERT(AC.cols() == n);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            RC_ASSERT(approxEqual(AC(i, j), A(i, j) + C(i, j)));
        }
    }

    // Test 5: Element-wise Multiplication
    // (A ⊙ C)[i][j] = A[i][j] * C[i][j]
    Matrix AelemC = A.elementWiseMultiply(C);
    RC_ASSERT(AelemC.rows() == m);
    RC_ASSERT(AelemC.cols() == n);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            RC_ASSERT(approxEqual(AelemC(i, j), A(i, j) * C(i, j)));
        }
    }
}

// **Validates: Requirements 9.3**
// Feature: neural-network-framework, Property 34: Matrix Transpose Involution
// Verify that for any matrix M, transposing twice returns the original matrix: (M^T)^T = M
RC_GTEST_PROP(MatrixPropertyTest, MatrixTransposeInvolution, ()) {
    // Generate random dimensions
    auto rows = *rc::gen::inRange<size_t>(1, 15);
    auto cols = *rc::gen::inRange<size_t>(1, 15);

    // Generate a random matrix
    auto M = *arbMatrixOfSize(rows, cols);

    // Compute (M^T)^T
    Matrix MT = M.transpose();
    Matrix MTT = MT.transpose();

    // Verify dimensions are preserved
    RC_ASSERT(MTT.rows() == M.rows());
    RC_ASSERT(MTT.cols() == M.cols());

    // Verify all elements are equal
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            RC_ASSERT(approxEqual(MTT(i, j), M(i, j)));
        }
    }
}

// **Validates: Requirements 9.3**
// Feature: neural-network-framework, Property 36: Matrix Multiplication Associativity
// Verify that for any matrices A, B, C with compatible dimensions, (A*B)*C = A*(B*C)
RC_GTEST_PROP(MatrixPropertyTest, MatrixMultiplicationAssociativity, ()) {
    // Generate random dimensions that are compatible for multiplication
    auto m = *rc::gen::inRange<size_t>(1, 8);
    auto n = *rc::gen::inRange<size_t>(1, 8);
    auto p = *rc::gen::inRange<size_t>(1, 8);
    auto q = *rc::gen::inRange<size_t>(1, 8);

    // Generate three matrices: A(m×n), B(n×p), C(p×q)
    auto A = *arbMatrixOfSize(m, n);
    auto B = *arbMatrixOfSize(n, p);
    auto C = *arbMatrixOfSize(p, q);

    // Compute (A*B)*C
    Matrix AB = A * B;
    Matrix AB_C = AB * C;

    // Compute A*(B*C)
    Matrix BC = B * C;
    Matrix A_BC = A * BC;

    // Verify dimensions match
    RC_ASSERT(AB_C.rows() == A_BC.rows());
    RC_ASSERT(AB_C.cols() == A_BC.cols());
    RC_ASSERT(AB_C.rows() == m);
    RC_ASSERT(AB_C.cols() == q);

    // Verify all elements are equal within numerical precision
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < q; ++j) {
            RC_ASSERT(approxEqual(AB_C(i, j), A_BC(i, j), 1e-6));
        }
    }
}

// Additional property: Matrix addition commutativity
// A + B = B + A
RC_GTEST_PROP(MatrixPropertyTest, MatrixAdditionCommutativity, ()) {
    auto rows = *rc::gen::inRange<size_t>(1, 10);
    auto cols = *rc::gen::inRange<size_t>(1, 10);

    auto A = *arbMatrixOfSize(rows, cols);
    auto B = *arbMatrixOfSize(rows, cols);

    Matrix APlusB = A + B;
    Matrix BPlusA = B + A;

    RC_ASSERT(APlusB.rows() == BPlusA.rows());
    RC_ASSERT(APlusB.cols() == BPlusA.cols());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            RC_ASSERT(approxEqual(APlusB(i, j), BPlusA(i, j)));
        }
    }
}

// Additional property: Matrix addition associativity
// (A + B) + C = A + (B + C)
RC_GTEST_PROP(MatrixPropertyTest, MatrixAdditionAssociativity, ()) {
    auto rows = *rc::gen::inRange<size_t>(1, 10);
    auto cols = *rc::gen::inRange<size_t>(1, 10);

    auto A = *arbMatrixOfSize(rows, cols);
    auto B = *arbMatrixOfSize(rows, cols);
    auto C = *arbMatrixOfSize(rows, cols);

    Matrix left = (A + B) + C;
    Matrix right = A + (B + C);

    RC_ASSERT(left.rows() == right.rows());
    RC_ASSERT(left.cols() == right.cols());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            RC_ASSERT(approxEqual(left(i, j), right(i, j), 1e-6));
        }
    }
}

// Additional property: Element-wise multiplication commutativity
// A ⊙ B = B ⊙ A
RC_GTEST_PROP(MatrixPropertyTest, ElementWiseMultiplicationCommutativity, ()) {
    auto rows = *rc::gen::inRange<size_t>(1, 10);
    auto cols = *rc::gen::inRange<size_t>(1, 10);

    auto A = *arbMatrixOfSize(rows, cols);
    auto B = *arbMatrixOfSize(rows, cols);

    Matrix AelemB = A.elementWiseMultiply(B);
    Matrix BelemA = B.elementWiseMultiply(A);

    RC_ASSERT(AelemB.rows() == BelemA.rows());
    RC_ASSERT(AelemB.cols() == BelemA.cols());

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            RC_ASSERT(approxEqual(AelemB(i, j), BelemA(i, j)));
        }
    }
}

// Additional property: Transpose of product
// (A*B)^T = B^T * A^T
RC_GTEST_PROP(MatrixPropertyTest, TransposeOfProduct, ()) {
    auto m = *rc::gen::inRange<size_t>(1, 8);
    auto n = *rc::gen::inRange<size_t>(1, 8);
    auto p = *rc::gen::inRange<size_t>(1, 8);

    auto A = *arbMatrixOfSize(m, n);
    auto B = *arbMatrixOfSize(n, p);

    // Compute (A*B)^T
    Matrix AB = A * B;
    Matrix AB_T = AB.transpose();

    // Compute B^T * A^T
    Matrix BT = B.transpose();
    Matrix AT = A.transpose();
    Matrix BT_AT = BT * AT;

    RC_ASSERT(AB_T.rows() == BT_AT.rows());
    RC_ASSERT(AB_T.cols() == BT_AT.cols());

    for (size_t i = 0; i < p; ++i) {
        for (size_t j = 0; j < m; ++j) {
            RC_ASSERT(approxEqual(AB_T(i, j), BT_AT(i, j), 1e-6));
        }
    }
}

int main(int argc, char** argv) {
    // Configure RapidCheck
    // Minimum 100 iterations as specified in design document
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
