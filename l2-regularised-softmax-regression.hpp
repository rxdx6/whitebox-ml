#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct SoftmaxRegressionResult {
  std::vector<std::vector<float>> weights;
  std::vector<float> biases;
};

inline std::vector<float>
predict(const std::vector<float> &features,
        const std::vector<std::vector<float>> &weights,
        const std::vector<float> &biases) {
  size_t num_classes = weights.size();
  std::vector<float> logits(num_classes);
  float maximum_logit = -INFINITY;
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    float dot_product = 0.0f;
    for (size_t feature_index = 0; feature_index < features.size();
         feature_index++) {
      dot_product +=
          weights[class_index][feature_index] * features[feature_index];
    }
    logits[class_index] = dot_product + biases[class_index];
    if (logits[class_index] > maximum_logit) {
      maximum_logit = logits[class_index];
    }
  }
  std::vector<float> probabilities(num_classes);
  float sum_of_exponentials = 0;
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    probabilities[class_index] = exp(logits[class_index] - maximum_logit);
    sum_of_exponentials += probabilities[class_index];
  }
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    probabilities[class_index] /= sum_of_exponentials;
  }
  return probabilities;
}

inline SoftmaxRegressionResult
update_w_and_b(const std::vector<std::vector<float>> &features_matrix,
               const std::vector<float> &target_labels,
               const std::vector<std::vector<float>> &weights,
               const std::vector<float> &biases, float learning_rate,
               float l2_regularisation_strength) {
  size_t num_classes = weights.size();
  size_t num_features = weights[0].size();
  float num_samples = features_matrix.size();
  std::vector<std::vector<float>> weight_gradients(
      num_classes, std::vector<float>(num_features, 0.0f));
  std::vector<float> bias_gradients(num_classes, 0.0f);

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    std::vector<float> predicted_probabilities =
        predict(features_matrix[sample_index], weights, biases);
    int true_class_index = static_cast<int>(target_labels[sample_index]);
    for (size_t class_index = 0; class_index < num_classes; class_index++) {
      float is_target_class =
          (true_class_index == static_cast<int>(class_index)) ? 1.0f : 0.0f;
      float prediction_error =
          predicted_probabilities[class_index] - is_target_class;
      for (size_t feature_index = 0; feature_index < num_features;
           feature_index++) {
        weight_gradients[class_index][feature_index] +=
            prediction_error * features_matrix[sample_index][feature_index];
      }
      bias_gradients[class_index] += prediction_error;
    }
  }

  std::vector<std::vector<float>> updated_weights = weights;
  std::vector<float> updated_biases = biases;
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    for (size_t feature_index = 0; feature_index < num_features;
         feature_index++) {
      updated_weights[class_index][feature_index] -=
          learning_rate *
          (((1.0f / num_samples) *
            weight_gradients[class_index][feature_index]) +
           (l2_regularisation_strength * weights[class_index][feature_index]));
    }
    updated_biases[class_index] -=
        learning_rate * ((1.0f / num_samples) * bias_gradients[class_index]);
  }

  return {updated_weights, updated_biases};
}

inline float avg_loss(const std::vector<std::vector<float>> &features_matrix,
                      const std::vector<float> &target_labels,
                      const std::vector<std::vector<float>> &weights,
                      const std::vector<float> &biases,
                      float l2_regularisation_strength) {
  float num_samples = features_matrix.size();
  float total_loss = 0;
  float epsilon = 1e-7;

  for (size_t sample_index = 0; sample_index < num_samples; sample_index++) {
    std::vector<float> predicted_probabilities =
        predict(features_matrix[sample_index], weights, biases);
    int true_class_index = static_cast<int>(target_labels[sample_index]);
    float predicted_probability = predicted_probabilities[true_class_index];
    predicted_probability =
        std::max(epsilon, std::min(1.0f - epsilon, predicted_probability));
    total_loss += -log(predicted_probability);
  }

  float weights_squared_sum = 0;
  for (const auto &weight_row : weights) {
    for (float weight_value : weight_row) {
      weights_squared_sum += weight_value * weight_value;
    }
  }
  float l2_penalty_term =
      0.5f * l2_regularisation_strength * weights_squared_sum;

  return (total_loss / num_samples) + l2_penalty_term;
}

inline SoftmaxRegressionResult
train(const std::vector<std::vector<float>> &features_matrix,
      const std::vector<float> &target_labels,
      const std::vector<std::vector<float>> &initial_weights,
      const std::vector<float> &initial_biases, float learning_rate,
      float l2_regularisation_strength, int num_epochs) {
  std::vector<std::vector<float>> current_weights = initial_weights;
  std::vector<float> current_biases = initial_biases;
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    SoftmaxRegressionResult regression_result = update_w_and_b(
        features_matrix, target_labels, current_weights, current_biases,
        learning_rate, l2_regularisation_strength);
    current_weights = regression_result.weights;
    current_biases = regression_result.biases;

    if (epoch % 1000 == 0) {
      float current_loss =
          avg_loss(features_matrix, target_labels, current_weights,
                   current_biases, l2_regularisation_strength);
      printf("Epoch %d - Current Log Loss: %f\n", epoch, current_loss);
    }
  }
  printf("Training complete. Final Log Loss: %f\n",
         avg_loss(features_matrix, target_labels, current_weights,
                  current_biases, l2_regularisation_strength));
  return {current_weights, current_biases};
}
