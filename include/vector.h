#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <initializer_list>
#include <cstddef>

class Vector {
public:
    // Constructors
    explicit Vector(size_t size);
    Vector(size_t size, double init_value);
    Vector(std::initializer_list<double> values);

    // Accessors
    double& operator[](size_t index);
    double operator[](size_t index) const;
    size_t size() const;

    // Operations
    Vector operator+(const Vector& other) const;
    Vector operator-(const Vector& other) const;
    Vector elementWiseMultiply(const Vector& other) const;
    double dot(const Vector& other) const;

    // Utilities
    void fill(double value);
    double sum() const;
    double mean() const;
    void print(const char* name = nullptr) const;  // Print vector to console

private:
    std::vector<double> data_;
};

#endif // VECTOR_H
