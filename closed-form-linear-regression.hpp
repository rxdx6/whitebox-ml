#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

inline std::vector<std::vector<float>>
matrix_transpose(const std::vector<std::vector<float>> &input_matrix) {
  if (input_matrix.empty() || input_matrix[0].empty()) {
    return {};
  }

  size_t rows = input_matrix.size();
  size_t columns = input_matrix[0].size();

  std::vector<std::vector<float>> transposed(columns,
                                             std::vector<float>(rows, 0));
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < columns; j++) {
      transposed[j][i] = input_matrix[i][j];
    }
  }

  return transposed;
}

inline std::vector<std::vector<float>>
matrix_multiplication(const std::vector<std::vector<float>> &matrix1,
                      const std::vector<std::vector<float>> &matrix2) {
  if (matrix1.empty() || matrix1[0].empty() || matrix2.empty() ||
      matrix2[0].empty()) {
    throw std::invalid_argument("Matrices cannot be empty for multiplication");
  }

  size_t r1 = matrix1.size();
  size_t c1 = matrix1[0].size();
  size_t r2 = matrix2.size();
  size_t c2 = matrix2[0].size();

  if (c1 != r2) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication");
  }

  std::vector<std::vector<float>> result(r1, std::vector<float>(c2, 0));

  for (size_t i = 0; i < r1; i++) {
    for (size_t j = 0; j < c2; j++) {
      for (size_t k = 0; k < c1; k++) {
        result[i][j] += matrix1[i][k] * matrix2[k][j];
      }
    }
  }

  return result;
}

inline std::vector<float>
solve_linear_system(const std::vector<std::vector<float>> &matrix1,
                    const std::vector<std::vector<float>> &matrix2) {
  if (matrix1.empty() || matrix2.empty()) {
    throw std::invalid_argument("System matrices cannot be empty");
  }

  size_t n = matrix1.size();
  std::vector<std::vector<float>> augmented_matrix(
      n, std::vector<float>(n + 1, 0));

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      augmented_matrix[i][j] = matrix1[i][j];
    }
    augmented_matrix[i][n] = matrix2[i][0];
  }

  for (size_t i = 0; i < n; i++) {
    size_t max_row = i;
    for (size_t k = i + 1; k < n; k++) {
      if (std::abs(augmented_matrix[k][i]) >
          std::abs(augmented_matrix[max_row][i])) {
        max_row = k;
      }
    }

    std::swap(augmented_matrix[i], augmented_matrix[max_row]);

    float pivot = augmented_matrix[i][i];

    if (std::abs(pivot) < 1e-7f) {
      throw std::runtime_error(
          "Matrix is singular or near-singular; system cannot be solved.");
    }

    for (size_t j = i; j <= n; j++) {
      augmented_matrix[i][j] /= pivot;
    }

    for (size_t k = 0; k < n; k++) {
      if (k != i) {
        float factor = augmented_matrix[k][i];
        for (size_t j = i; j <= n; j++) {
          augmented_matrix[k][j] -= factor * augmented_matrix[i][j];
        }
      }
    }
  }

  std::vector<float> w(n);
  for (size_t i = 0; i < n; i++) {
    w[i] = augmented_matrix[i][n];
  }

  return w;
}

inline std::vector<float> linear_regressor(std::vector<std::vector<float>> X,
                                           const std::vector<float> &y) {
  if (X.empty() || y.empty() || X.size() != y.size()) {
    throw std::invalid_argument(
        "Sample counts in X and y must match and not be empty");
  }

  size_t number_of_samples = X.size();

  for (size_t row = 0; row < X.size(); row++) {
    X[row].push_back(1.0f);
  }

  std::vector<std::vector<float>> y_matrix(number_of_samples,
                                           std::vector<float>(1));
  for (size_t i = 0; i < number_of_samples; i++) {
    y_matrix[i][0] = y[i];
  }

  std::vector<std::vector<float>> Xt = matrix_transpose(X);
  std::vector<std::vector<float>> XtX = matrix_multiplication(Xt, X);
  std::vector<std::vector<float>> XtY = matrix_multiplication(Xt, y_matrix);

  std::vector<float> w = solve_linear_system(XtX, XtY);

  return w;
}
