#include "matrix.h"
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

// Constructors
Matrix::Matrix(size_t rows, size_t cols)
    : data_(rows * cols, 0.0), rows_(rows), cols_(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Matrix dimensions must be greater than zero. "
            "Provided: rows=" + std::to_string(rows) +
            ", cols=" + std::to_string(cols)
        );
    }
}

Matrix::Matrix(size_t rows, size_t cols, double init_value)
    : data_(rows * cols, init_value), rows_(rows), cols_(cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument(
            "Matrix dimensions must be greater than zero. "
            "Provided: rows=" + std::to_string(rows) +
            ", cols=" + std::to_string(cols)
        );
    }
}

// Accessors
double& Matrix::operator()(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

double Matrix::operator()(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}

size_t Matrix::rows() const {
    return rows_;
}

size_t Matrix::cols() const {
    return cols_;
}

// Operations
Matrix Matrix::transpose() const {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrix addition failed: dimensions mismatch. "
            "Left matrix: " + std::to_string(rows_) + "x" + std::to_string(cols_) +
            ", Right matrix: " + std::to_string(other.rows_) + "x" + std::to_string(other.cols_)
        );
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument(
            "Matrix multiplication failed: incompatible dimensions. "
            "Left matrix: " + std::to_string(rows_) + "x" + std::to_string(cols_) +
            ", Right matrix: " + std::to_string(other.rows_) + "x" + std::to_string(other.cols_) +
            ". Expected: left.cols == right.rows"
        );
    }

    Matrix result(rows_, other.cols_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols_; ++k) {
                sum += (*this)(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Vector Matrix::operator*(const Vector& vec) const {
    if (cols_ != vec.size()) {
        throw std::invalid_argument(
            "Matrix-vector multiplication failed: incompatible dimensions. "
            "Matrix: " + std::to_string(rows_) + "x" + std::to_string(cols_) +
            ", Vector size: " + std::to_string(vec.size()) +
            ". Expected: matrix.cols == vector.size"
        );
    }

    Vector result(rows_);
    for (size_t i = 0; i < rows_; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < cols_; ++j) {
            sum += (*this)(i, j) * vec[j];
        }
        result[i] = sum;
    }
    return result;
}

Matrix Matrix::elementWiseMultiply(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument(
            "Matrix element-wise multiplication failed: dimensions mismatch. "
            "Left matrix: " + std::to_string(rows_) + "x" + std::to_string(cols_) +
            ", Right matrix: " + std::to_string(other.rows_) + "x" + std::to_string(other.cols_)
        );
    }

    Matrix result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// Utilities
void Matrix::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::randomize(double min, double max) {
    if (min > max) {
        throw std::invalid_argument(
            "Matrix randomize failed: min must be less than or equal to max. "
            "Provided: min=" + std::to_string(min) + ", max=" + std::to_string(max)
        );
    }

    std::uniform_real_distribution<double> dist(min, max);
    std::mt19937& gen = getGenerator();

    for (double& value : data_) {
        value = dist(gen);
    }
}

// Static member initialization
bool Matrix::seed_set_ = false;
unsigned int Matrix::seed_value_ = 0;

/**
 * Set the random seed for weight initialization.
 * Call this before initializeXavier/initializeHe/randomize for reproducible results.
 *
 * @param seed The seed value for the random number generator
 */
void Matrix::setSeed(unsigned int seed) {
    seed_set_ = true;
    seed_value_ = seed;
    // Reset the generator with the new seed
    getGenerator().seed(seed);
}

/**
 * Get the internal random number generator.
 * If setSeed() was called, uses that seed. Otherwise uses std::random_device.
 */
std::mt19937& Matrix::getGenerator() {
    static std::mt19937 gen(std::random_device{}());
    return gen;
}

void Matrix::print(const char* name) const {
    if (name != nullptr) {
        std::cout << name << " (" << rows_ << "x" << cols_ << "):" << std::endl;
    }

    // Find max width for alignment
    size_t max_width = 0;
    for (const auto& val : data_) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(4) << val;
        max_width = std::max(max_width, oss.str().length());
    }

    // Print matrix with aligned columns
    for (size_t i = 0; i < rows_; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols_; ++j) {
            std::cout << std::setw(max_width + 1) << std::fixed
                      << std::setprecision(4) << (*this)(i, j);
            if (j < cols_ - 1) {
                std::cout << " ";
            }
        }
        std::cout << " ]" << std::endl;
    }

    if (name == nullptr) {
        std::cout << "(" << rows_ << "x" << cols_ << ")" << std::endl;
    }
}
