#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

inline float standardise(float val, float mean, float standard_deviation) {
  return (standard_deviation == 0.0f) ? 0.0f
                                      : (val - mean) / standard_deviation;
}

inline float unstandardise(float standardised_val, float mean,
                           float standard_deviation) {
  return (standardised_val * standard_deviation) + mean;
}

struct StandardiseResult {
  std::vector<float> scaled_vector;
  float mean;
  float std_dev;
};

inline StandardiseResult standardise(const std::vector<float> &v) {
  if (v.empty()) {
    return {{}, 0.0f, 1.0f};
  }

  float sum = std::accumulate(v.begin(), v.end(), 0.0f);
  float average = sum / v.size();

  float variance = 0.0f;
  for (const float val : v) {
    float diff = val - average;
    variance += diff * diff;
  }
  float standard_deviation = std::sqrt(variance / v.size());

  if (standard_deviation == 0.0f) {
    return {std::vector<float>(v.size(), 0.0f), average, 0.0f};
  }

  std::vector<float> res;
  res.reserve(v.size());
  for (const float val : v) {
    res.push_back(standardise(val, average, standard_deviation));
  }

  return {res, average, standard_deviation};
}

struct StandardiseMatrixResult {
  std::vector<std::vector<float>> scaled_matrix;
  std::vector<float> means;
  std::vector<float> std_devs;
};

inline StandardiseMatrixResult standardise_matrix(const std::vector<std::vector<float>> &X) {
  if (X.empty()) {
    return {{}, {}, {}};
  }
  size_t num_samples = X.size();
  size_t num_features = X[0].size();
  std::vector<std::vector<float>> scaled_matrix(num_samples, std::vector<float>(num_features));
  std::vector<float> means(num_features);
  std::vector<float> std_devs(num_features);

  for (size_t j = 0; j < num_features; ++j) {
    std::vector<float> col(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
      col[i] = X[i][j];
    }
    auto [col_scaled, col_mean, col_std] = standardise(col);
    means[j] = col_mean;
    std_devs[j] = col_std;
    for (size_t i = 0; i < num_samples; ++i) {
      scaled_matrix[i][j] = col_scaled[i];
    }
  }

  return {scaled_matrix, means, std_devs};
}

inline std::vector<float> standardise_vector(const std::vector<float> &v,
                                             const std::vector<float> &means,
                                             const std::vector<float> &std_devs) {
  std::vector<float> scaled_v(v.size());
  for (size_t j = 0; j < v.size(); ++j) {
    scaled_v[j] = standardise(v[j], means[j], std_devs[j]);
  }
  return scaled_v;
}
