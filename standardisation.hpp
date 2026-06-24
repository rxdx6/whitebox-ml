#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

inline float standardise(float value, float mean, float standard_deviation) {
  return (standard_deviation == 0.0f) ? 0.0f
                                      : (value - mean) / standard_deviation;
}

inline float unstandardise(float standardised_value, float mean,
                           float standard_deviation) {
  return (standardised_value * standard_deviation) + mean;
}

struct StandardiseResult {
  std::vector<float> scaled_vector;
  float mean;
  float standard_deviation;
};

inline StandardiseResult standardise(const std::vector<float> &x) {
  if (x.empty()) {
    return {{}, 0.0f, 1.0f};
  }

  float sum = std::accumulate(x.begin(), x.end(), 0.0f);
  float average = sum / x.size();

  float variance = 0.0f;
  for (const float value : x) {
    float difference = value - average;
    variance += difference * difference;
  }
  float standard_deviation = std::sqrt(variance / x.size());

  if (standard_deviation == 0.0f) {
    return {std::vector<float>(x.size(), 0.0f), average, 0.0f};
  }

  std::vector<float> scaled_vector;
  scaled_vector.reserve(x.size());
  for (const float value : x) {
    scaled_vector.push_back(standardise(value, average, standard_deviation));
  }

  return {scaled_vector, average, standard_deviation};
}

struct StandardiseMatrixResult {
  std::vector<std::vector<float>> scaled_matrix;
  std::vector<float> means;
  std::vector<float> standard_deviations;
};

inline StandardiseMatrixResult
standardise_matrix(const std::vector<std::vector<float>> &X) {
  if (X.empty()) {
    return {{}, {}, {}};
  }
  size_t m = X.size();
  size_t d = X[0].size();
  std::vector<std::vector<float>> scaled_matrix(m, std::vector<float>(d));
  std::vector<float> means(d);
  std::vector<float> standard_deviations(d);

  for (size_t j = 0; j < d; ++j) {
    std::vector<float> column_vector(m);
    for (size_t i = 0; i < m; ++i) {
      column_vector[i] = X[i][j];
    }
    auto [scaled_column, column_mean, column_standard_deviation] =
        standardise(column_vector);
    means[j] = column_mean;
    standard_deviations[j] = column_standard_deviation;
    for (size_t i = 0; i < m; ++i) {
      scaled_matrix[i][j] = scaled_column[i];
    }
  }

  return {scaled_matrix, means, standard_deviations};
}

inline std::vector<float>
standardise_vector(const std::vector<float> &x,
                   const std::vector<float> &means,
                   const std::vector<float> &standard_deviations) {
  std::vector<float> scaled_vector(x.size());
  for (size_t j = 0; j < x.size(); ++j) {
    scaled_vector[j] =
        standardise(x[j], means[j], standard_deviations[j]);
  }
  return scaled_vector;
}
