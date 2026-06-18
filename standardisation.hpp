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

inline StandardiseResult standardise(const std::vector<float> &input_vector) {
  if (input_vector.empty()) {
    return {{}, 0.0f, 1.0f};
  }

  float sum = std::accumulate(input_vector.begin(), input_vector.end(), 0.0f);
  float average = sum / input_vector.size();

  float variance = 0.0f;
  for (const float value : input_vector) {
    float difference = value - average;
    variance += difference * difference;
  }
  float standard_deviation = std::sqrt(variance / input_vector.size());

  if (standard_deviation == 0.0f) {
    return {std::vector<float>(input_vector.size(), 0.0f), average, 0.0f};
  }

  std::vector<float> scaled_vector;
  scaled_vector.reserve(input_vector.size());
  for (const float value : input_vector) {
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
standardise_matrix(const std::vector<std::vector<float>> &features_matrix) {
  if (features_matrix.empty()) {
    return {{}, {}, {}};
  }
  size_t num_samples = features_matrix.size();
  size_t num_features = features_matrix[0].size();
  std::vector<std::vector<float>> scaled_matrix(
      num_samples, std::vector<float>(num_features));
  std::vector<float> means(num_features);
  std::vector<float> standard_deviations(num_features);

  for (size_t feature_index = 0; feature_index < num_features;
       ++feature_index) {
    std::vector<float> column_vector(num_samples);
    for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
      column_vector[sample_index] =
          features_matrix[sample_index][feature_index];
    }
    auto [scaled_column, column_mean, column_standard_deviation] =
        standardise(column_vector);
    means[feature_index] = column_mean;
    standard_deviations[feature_index] = column_standard_deviation;
    for (size_t sample_index = 0; sample_index < num_samples; ++sample_index) {
      scaled_matrix[sample_index][feature_index] = scaled_column[sample_index];
    }
  }

  return {scaled_matrix, means, standard_deviations};
}

inline std::vector<float>
standardise_vector(const std::vector<float> &input_vector,
                   const std::vector<float> &means,
                   const std::vector<float> &standard_deviations) {
  std::vector<float> scaled_vector(input_vector.size());
  for (size_t feature_index = 0; feature_index < input_vector.size();
       ++feature_index) {
    scaled_vector[feature_index] =
        standardise(input_vector[feature_index], means[feature_index],
                    standard_deviations[feature_index]);
  }
  return scaled_vector;
}
