#pragma once

#include <vector>
inline std::vector<float> update_w_and_b(const std::vector<float> &features,
                                         const std::vector<float> &target_labels, float weight,
                                         float bias, float learning_rate,
                                         float l2_regularisation_strength) {
  float weight_gradient = 0;
  float bias_gradient = 0;
  float num_samples = features.size();

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    if (target_labels[sample_index] * ((weight * features[sample_index]) + bias) < 1) {
      weight_gradient += -target_labels[sample_index] * features[sample_index];
      bias_gradient += -target_labels[sample_index];
    }
  }

  weight -= learning_rate * (l2_regularisation_strength * weight + (weight_gradient / num_samples));
  bias -= learning_rate * (bias_gradient / num_samples);

  return {weight, bias};
}

inline float avg_loss(const std::vector<float> &features, const std::vector<float> &target_labels,
                      float weight, float bias, float l2_regularisation_strength) {
  float num_samples = features.size();
  float total_hinge_loss = 0;

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    float margin_distance = 1 - target_labels[sample_index] * ((weight * features[sample_index]) + bias);
    if (margin_distance > 0) {
      total_hinge_loss += margin_distance;
    }
  }

  float l2_penalty_term = 0.5 * l2_regularisation_strength * weight * weight;
  return l2_penalty_term + (total_hinge_loss / num_samples);
}

inline std::vector<float> train(const std::vector<float> &features,
                                const std::vector<float> &target_labels, float weight, float bias,
                                float learning_rate, float l2_regularisation_strength, int num_epochs) {
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<float> updated_weight_and_bias =
        update_w_and_b(features, target_labels, weight, bias, learning_rate, l2_regularisation_strength);
    weight = updated_weight_and_bias[0];
    bias = updated_weight_and_bias[1];
  }

  return {weight, bias};
}

inline float predict(float feature_value, float weight, float bias) {
  float raw_score = (weight * feature_value) + bias;
  return (raw_score >= 0) ? 1 : -1;
}
