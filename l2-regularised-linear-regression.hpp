#pragma once

#include <vector>

struct LinearRegressionResult {
  std::vector<float> weights;
  float bias;
};

inline LinearRegressionResult
update_w_and_b(const std::vector<std::vector<float>> &features_matrix,
               const std::vector<float> &target_labels,
               const std::vector<float> &weights, float bias,
               float learning_rate, float l2_regularisation_strength) {
  float bias_gradient = 0.0f;
  float num_samples = features_matrix.size();
  size_t num_features = weights.size();
  std::vector<float> weight_gradients(num_features, 0.0f);

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    float predicted_value = bias;
    for (size_t feature_index = 0; feature_index < num_features;
         feature_index++) {
      predicted_value +=
          weights[feature_index] * features_matrix[sample_index][feature_index];
    }

    float prediction_error = target_labels[sample_index] - predicted_value;
    for (size_t feature_index = 0; feature_index < num_features;
         feature_index++) {
      weight_gradients[feature_index] +=
          -2 * features_matrix[sample_index][feature_index] * prediction_error;
    }
    bias_gradient += -2 * prediction_error;
  }

  std::vector<float> updated_weights = weights;
  for (size_t feature_index = 0; feature_index < num_features;
       feature_index++) {
    updated_weights[feature_index] -=
        learning_rate *
        (((1.0f / num_samples) * weight_gradients[feature_index]) +
         (l2_regularisation_strength * weights[feature_index]));
  }
  float updated_bias =
      bias - (1.0f / num_samples) * bias_gradient * learning_rate;

  return {updated_weights, updated_bias};
}

inline float avg_loss(const std::vector<std::vector<float>> &features_matrix,
                      const std::vector<float> &target_labels,
                      const std::vector<float> &weights, float bias,
                      float l2_regularisation_strength) {
  float num_samples = features_matrix.size();
  float total_error = 0.0f;
  size_t num_features = weights.size();

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    float predicted_value = bias;
    for (size_t feature_index = 0; feature_index < num_features;
         feature_index++) {
      predicted_value +=
          weights[feature_index] * features_matrix[sample_index][feature_index];
    }
    float error = target_labels[sample_index] - predicted_value;
    total_error += error * error;
  }

  float mean_squared_error = total_error / num_samples;
  float weights_squared_sum = 0.0f;
  for (float weight_value : weights) {
    weights_squared_sum += weight_value * weight_value;
  }
  float l2_penalty_term =
      0.5f * l2_regularisation_strength * weights_squared_sum;

  return mean_squared_error + l2_penalty_term;
}

inline LinearRegressionResult
train(const std::vector<std::vector<float>> &features_matrix,
      const std::vector<float> &target_labels,
      const std::vector<float> &weights, float bias, float learning_rate,
      float l2_regularisation_strength, int num_epochs) {
  std::vector<float> current_weights = weights;
  float current_bias = bias;
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    LinearRegressionResult regression_result =
        update_w_and_b(features_matrix, target_labels, current_weights,
                       current_bias, learning_rate, l2_regularisation_strength);
    current_weights = regression_result.weights;
    current_bias = regression_result.bias;
  }

  return {current_weights, current_bias};
}

inline std::vector<float>
predict(const std::vector<std::vector<float>> &features_matrix,
        const std::vector<float> &weights, float bias) {
  std::vector<float> predictions(features_matrix.size());
  for (size_t sample_index = 0; sample_index < features_matrix.size();
       sample_index++) {
    float predicted_value = bias;
    for (size_t feature_index = 0; feature_index < weights.size();
         feature_index++) {
      predicted_value +=
          weights[feature_index] * features_matrix[sample_index][feature_index];
    }
    predictions[sample_index] = predicted_value;
  }
  return predictions;
}
