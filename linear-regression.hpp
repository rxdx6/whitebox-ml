#pragma once

#include <vector>
inline std::vector<float> update_w_and_b(const std::vector<float> &features,
                                         const std::vector<float> &target_labels, float weight,
                                         float bias, float learning_rate) {
  float weight_gradient = 0;
  float bias_gradient = 0;
  float num_samples = features.size();

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    weight_gradient += -2 * features[sample_index] * (target_labels[sample_index] - (weight * features[sample_index] + bias));
    bias_gradient += -2 * (target_labels[sample_index] - (weight * features[sample_index] + bias));
  }

  weight -= (1 / num_samples) * weight_gradient * learning_rate;
  bias -= (1 / num_samples) * bias_gradient * learning_rate;

  return {weight, bias};
}

inline float avg_loss(const std::vector<float> &features, const std::vector<float> &target_labels,
                      float weight, float bias) {
  float num_samples = features.size();
  float total_error = 0;

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    total_error += (target_labels[sample_index] - (weight * features[sample_index] + bias)) * (target_labels[sample_index] - (weight * features[sample_index] + bias));
  }

  return total_error / num_samples;
}

inline std::vector<float> train(const std::vector<float> &features,
                                const std::vector<float> &target_labels, float weight, float bias,
                                float learning_rate, int num_epochs) {
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<float> updated_weight_and_bias = update_w_and_b(features, target_labels, weight, bias, learning_rate);
    weight = updated_weight_and_bias[0];
    bias = updated_weight_and_bias[1];
  }

  return {weight, bias};
}

inline float predict(float feature_value, float weight, float bias) { return (weight * feature_value) + bias; }
