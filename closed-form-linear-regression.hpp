#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

inline std::vector<std::vector<float>>
matrix_transpose(const std::vector<std::vector<float>> &A) {
  if (A.empty() || A[0].empty()) {
    return {};
  }

  size_t rows = A.size();
  size_t cols = A[0].size();

  std::vector<std::vector<float>> A_T(cols, std::vector<float>(rows, 0));
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      A_T[j][i] = A[i][j];
    }
  }

  return A_T;
}

inline std::vector<std::vector<float>>
matrix_multiplication(const std::vector<std::vector<float>> &A,
                      const std::vector<std::vector<float>> &B) {
  if (A.empty() || A[0].empty() || B.empty() || B[0].empty()) {
    throw std::invalid_argument("Matrices cannot be empty for multiplication");
  }

  size_t A_rows = A.size();
  size_t A_cols = A[0].size();
  size_t B_rows = B.size();
  size_t B_cols = B[0].size();

  if (A_cols != B_rows) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication");
  }

  std::vector<std::vector<float>> C(A_rows, std::vector<float>(B_cols, 0));

  for (size_t i = 0; i < A_rows; i++) {
    for (size_t j = 0; j < B_cols; j++) {
      for (size_t k = 0; k < A_cols; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return C;
}

inline std::vector<float>
solve_linear_system(const std::vector<std::vector<float>> &A,
                    const std::vector<std::vector<float>> &B) {
  if (A.empty() || B.empty()) {
    throw std::invalid_argument("System matrices cannot be empty");
  }

  size_t n = A.size();
  std::vector<std::vector<float>> augmented(n, std::vector<float>(n + 1, 0));

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      augmented[i][j] = A[i][j];
    }
    augmented[i][n] = B[i][0];
  }

  for (size_t k = 0; k < n; k++) {
    size_t max_row = k;
    for (size_t i = k + 1; i < n; i++) {
      if (std::abs(augmented[i][k]) > std::abs(augmented[max_row][k])) {
        max_row = i;
      }
    }

    std::swap(augmented[k], augmented[max_row]);

    float pivot = augmented[k][k];

    if (std::abs(pivot) < 1e-7f) {
      throw std::runtime_error(
          "Matrix is singular or near-singular; system cannot be solved.");
    }

    for (size_t j = k; j <= n; j++) {
      augmented[k][j] /= pivot;
    }

    for (size_t i = 0; i < n; i++) {
      if (i != k) {
        float factor = augmented[i][k];
        for (size_t j = k; j <= n; j++) {
          augmented[i][j] -= factor * augmented[k][j];
        }
      }
    }
  }

  std::vector<float> w(n);
  for (size_t i = 0; i < n; i++) {
    w[i] = augmented[i][n];
  }

  return w;
}

inline std::vector<float>
linear_regressor(std::vector<std::vector<float>> X,
                 const std::vector<float> &y) {
  if (X.empty() || y.empty() || X.size() != y.size()) {
    throw std::invalid_argument(
        "Sample counts in X and y must match and not be empty");
  }

  size_t m = X.size();

  for (size_t i = 0; i < X.size(); i++) {
    X[i].push_back(1.0f);
  }

  std::vector<std::vector<float>> Y(m, std::vector<float>(1));
  for (size_t i = 0; i < m; i++) {
    Y[i][0] = y[i];
  }

  std::vector<std::vector<float>> X_T = matrix_transpose(X);
  std::vector<std::vector<float>> X_T_X = matrix_multiplication(X_T, X);
  std::vector<std::vector<float>> X_T_Y = matrix_multiplication(X_T, Y);

  std::vector<float> w = solve_linear_system(X_T_X, X_T_Y);

  return w;
}
