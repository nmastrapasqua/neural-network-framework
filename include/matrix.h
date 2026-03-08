#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef>
#include "vector.h"

class Matrix {
public:
    // Constructors
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, double init_value);

    // Accessors
    double& operator()(size_t row, size_t col);
    double operator()(size_t row, size_t col) const;
    size_t rows() const;
    size_t cols() const;

    // Operations
    Matrix transpose() const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Vector operator*(const Vector& vec) const;
    Matrix elementWiseMultiply(const Matrix& other) const;

    // Utilities
    void fill(double value);
    void randomize(double min, double max);
    void print(const char* name = nullptr) const;  // Print matrix to console

private:
    std::vector<double> data_;  // Row-major order storage
    size_t rows_;
    size_t cols_;
};

#endif // MATRIX_H
