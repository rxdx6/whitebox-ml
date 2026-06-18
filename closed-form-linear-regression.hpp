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

  size_t num_rows = input_matrix.size();
  size_t num_columns = input_matrix[0].size();

  std::vector<std::vector<float>> transposed(num_columns,
                                             std::vector<float>(num_rows, 0));
  for (size_t row_index = 0; row_index < num_rows; row_index++) {
    for (size_t column_index = 0; column_index < num_columns; column_index++) {
      transposed[column_index][row_index] = input_matrix[row_index][column_index];
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

  size_t matrix1_rows = matrix1.size();
  size_t matrix1_columns = matrix1[0].size();
  size_t matrix2_rows = matrix2.size();
  size_t matrix2_columns = matrix2[0].size();

  if (matrix1_columns != matrix2_rows) {
    throw std::invalid_argument(
        "Matrix dimensions do not match for multiplication");
  }

  std::vector<std::vector<float>> result(matrix1_rows, std::vector<float>(matrix2_columns, 0));

  for (size_t row_index = 0; row_index < matrix1_rows; row_index++) {
    for (size_t column_index = 0; column_index < matrix2_columns; column_index++) {
      for (size_t inner_index = 0; inner_index < matrix1_columns; inner_index++) {
        result[row_index][column_index] += matrix1[row_index][inner_index] * matrix2[inner_index][column_index];
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

  size_t matrix_size = matrix1.size();
  std::vector<std::vector<float>> augmented_matrix(
      matrix_size, std::vector<float>(matrix_size + 1, 0));

  for (size_t row_index = 0; row_index < matrix_size; row_index++) {
    for (size_t column_index = 0; column_index < matrix_size; column_index++) {
      augmented_matrix[row_index][column_index] = matrix1[row_index][column_index];
    }
    augmented_matrix[row_index][matrix_size] = matrix2[row_index][0];
  }

  for (size_t pivot_index = 0; pivot_index < matrix_size; pivot_index++) {
    size_t max_row_index = pivot_index;
    for (size_t row_index = pivot_index + 1; row_index < matrix_size; row_index++) {
      if (std::abs(augmented_matrix[row_index][pivot_index]) >
          std::abs(augmented_matrix[max_row_index][pivot_index])) {
        max_row_index = row_index;
      }
    }

    std::swap(augmented_matrix[pivot_index], augmented_matrix[max_row_index]);

    float pivot_value = augmented_matrix[pivot_index][pivot_index];

    if (std::abs(pivot_value) < 1e-7f) {
      throw std::runtime_error(
          "Matrix is singular or near-singular; system cannot be solved.");
    }

    for (size_t column_index = pivot_index; column_index <= matrix_size; column_index++) {
      augmented_matrix[pivot_index][column_index] /= pivot_value;
    }

    for (size_t row_index = 0; row_index < matrix_size; row_index++) {
      if (row_index != pivot_index) {
        float elimination_factor = augmented_matrix[row_index][pivot_index];
        for (size_t column_index = pivot_index; column_index <= matrix_size; column_index++) {
          augmented_matrix[row_index][column_index] -= elimination_factor * augmented_matrix[pivot_index][column_index];
        }
      }
    }
  }

  std::vector<float> weights(matrix_size);
  for (size_t row_index = 0; row_index < matrix_size; row_index++) {
    weights[row_index] = augmented_matrix[row_index][matrix_size];
  }

  return weights;
}

inline std::vector<float> linear_regressor(std::vector<std::vector<float>> features_matrix,
                                           const std::vector<float> &target_labels) {
  if (features_matrix.empty() || target_labels.empty() || features_matrix.size() != target_labels.size()) {
    throw std::invalid_argument(
        "Sample counts in X and y must match and not be empty");
  }

  size_t num_samples = features_matrix.size();

  for (size_t sample_index = 0; sample_index < features_matrix.size(); sample_index++) {
    features_matrix[sample_index].push_back(1.0f);
  }

  std::vector<std::vector<float>> target_labels_matrix(num_samples,
                                           std::vector<float>(1));
  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    target_labels_matrix[sample_index][0] = target_labels[sample_index];
  }

  std::vector<std::vector<float>> transposed_features = matrix_transpose(features_matrix);
  std::vector<std::vector<float>> transposed_features_times_features = matrix_multiplication(transposed_features, features_matrix);
  std::vector<std::vector<float>> transposed_features_times_targets = matrix_multiplication(transposed_features, target_labels_matrix);

  std::vector<float> weights = solve_linear_system(transposed_features_times_features, transposed_features_times_targets);

  return weights;
}
