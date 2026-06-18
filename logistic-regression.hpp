#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

inline float sigmoid_function(float logit_value) { return 1 / (1 + exp(-logit_value)); }

inline float predict(float feature_value, float weight, float bias) {
  return sigmoid_function((weight * feature_value) + bias);
}

inline std::vector<float> update_w_and_b(const std::vector<float> &features,
                                         const std::vector<float> &target_labels, float weight,
                                         float bias, float learning_rate) {
  float weight_gradient = 0;
  float bias_gradient = 0;
  float num_samples = features.size();

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    float predicted_probability = predict(features[sample_index], weight, bias);
    weight_gradient += (predicted_probability - target_labels[sample_index]) * features[sample_index];
    bias_gradient += predicted_probability - target_labels[sample_index];
  }

  weight -= (1 / num_samples) * weight_gradient * learning_rate;
  bias -= (1 / num_samples) * bias_gradient * learning_rate;

  return {weight, bias};
}

inline float avg_loss(const std::vector<float> &features, const std::vector<float> &target_labels,
                      float weight, float bias) {
  float num_samples = features.size();
  float total_loss = 0;
  float epsilon = 1e-7;

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    float predicted_probability = predict(features[sample_index], weight, bias);
    predicted_probability = std::max(epsilon, std::min(1 - epsilon, predicted_probability));
    total_loss += -(target_labels[sample_index] * log(predicted_probability) + (1 - target_labels[sample_index]) * log(1 - predicted_probability));
  }

  return total_loss / num_samples;
}

inline std::vector<float> train(const std::vector<float> &features,
                                const std::vector<float> &target_labels, float weight, float bias,
                                float learning_rate, int num_epochs) {
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<float> updated_weight_and_bias = update_w_and_b(features, target_labels, weight, bias, learning_rate);
    weight = updated_weight_and_bias[0];
    bias = updated_weight_and_bias[1];

    if (epoch % 1000 == 0) {
      float current_loss = avg_loss(features, target_labels, weight, bias);
      printf("Epoch %d - Current Log Loss: %f\n", epoch, current_loss);
    }
  }
  printf("Training complete. Final Log Loss: %f\n", avg_loss(features, target_labels, weight, bias));
  return {weight, bias};
}
