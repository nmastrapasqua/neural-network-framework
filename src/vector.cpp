#include "vector.h"
#include <stdexcept>
#include <numeric>
#include <iostream>
#include <iomanip>

// Constructors
Vector::Vector(size_t size) : data_(size, 0.0) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero");
    }
}

Vector::Vector(size_t size, double init_value) : data_(size, init_value) {
    if (size == 0) {
        throw std::invalid_argument("Vector size must be greater than zero");
    }
}

Vector::Vector(std::initializer_list<double> values) : data_(values) {}

// Accessors
double& Vector::operator[](size_t index) {
    return data_[index];
}

double Vector::operator[](size_t index) const {
    return data_[index];
}

size_t Vector::size() const {
    return data_.size();
}

// Operations
Vector Vector::operator+(const Vector& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument(
            "Vector addition failed: dimensions mismatch. "
            "Left vector size: " + std::to_string(data_.size()) +
            ", Right vector size: " + std::to_string(other.data_.size())
        );
    }

    Vector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Vector Vector::operator-(const Vector& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument(
            "Vector subtraction failed: dimensions mismatch. "
            "Left vector size: " + std::to_string(data_.size()) +
            ", Right vector size: " + std::to_string(other.data_.size())
        );
    }

    Vector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Vector Vector::elementWiseMultiply(const Vector& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument(
            "Vector element-wise multiplication failed: dimensions mismatch. "
            "Left vector size: " + std::to_string(data_.size()) +
            ", Right vector size: " + std::to_string(other.data_.size())
        );
    }

    Vector result(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

double Vector::dot(const Vector& other) const {
    if (data_.size() != other.data_.size()) {
        throw std::invalid_argument(
            "Vector dot product failed: dimensions mismatch. "
            "Left vector size: " + std::to_string(data_.size()) +
            ", Right vector size: " + std::to_string(other.data_.size())
        );
    }

    double result = 0.0;
    for (size_t i = 0; i < data_.size(); ++i) {
        result += data_[i] * other.data_[i];
    }
    return result;
}

// Utilities
void Vector::fill(double value) {
    std::fill(data_.begin(), data_.end(), value);
}

double Vector::sum() const {
    return std::accumulate(data_.begin(), data_.end(), 0.0);
}

double Vector::mean() const {
    if (data_.empty()) {
        throw std::invalid_argument("Cannot compute mean of empty vector");
    }
    return sum() / static_cast<double>(data_.size());
}

void Vector::print(const char* name) const {
    if (name != nullptr) {
        std::cout << name << " = ";
    }

    std::cout << "[";
    for (size_t i = 0; i < data_.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << data_[i];
        if (i < data_.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]";

    if (name != nullptr) {
        std::cout << "  (size: " << data_.size() << ")";
    }

    std::cout << std::endl;
}
