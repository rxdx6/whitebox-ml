#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct SoftmaxRegressionResult {
  std::vector<float> w;
  std::vector<float> b;
};

inline std::vector<float> predict(const std::vector<float> &x,
                                  const std::vector<float> &w,
                                  const std::vector<float> &b) {
  size_t num_classes = b.size();
  size_t d = x.size();
  std::vector<float> logits(num_classes);
  float maximum_logit = -INFINITY;
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    float dot_product = 0.0f;
    for (size_t j = 0; j < d; j++) {
      dot_product += w[class_index * d + j] * x[j];
    }
    logits[class_index] = dot_product + b[class_index];
    if (logits[class_index] > maximum_logit) {
      maximum_logit = logits[class_index];
    }
  }
  std::vector<float> hat_y(num_classes);
  float sum_of_exponentials = 0;
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    hat_y[class_index] = exp(logits[class_index] - maximum_logit);
    sum_of_exponentials += hat_y[class_index];
  }
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    hat_y[class_index] /= sum_of_exponentials;
  }
  return hat_y;
}

inline SoftmaxRegressionResult
update_w_and_b(const std::vector<std::vector<float>> &X,
               const std::vector<float> &y,
               const std::vector<float> &w,
               const std::vector<float> &b, float alpha,
               float l2_regularisation_strength) {
  size_t num_classes = b.size();
  size_t d = X[0].size();
  float m = X.size();
  std::vector<float> dw(num_classes * d, 0.0f);
  std::vector<float> db(num_classes, 0.0f);

  for (size_t i = 0; i < m; i++) {
    std::vector<float> hat_y = predict(X[i], w, b);
    int true_class_index = static_cast<int>(y[i]);
    for (size_t class_index = 0; class_index < num_classes; class_index++) {
      float is_target_class =
          (true_class_index == static_cast<int>(class_index)) ? 1.0f : 0.0f;
      float prediction_error = hat_y[class_index] - is_target_class;
      for (size_t j = 0; j < d; j++) {
        dw[class_index * d + j] += prediction_error * X[i][j];
      }
      db[class_index] += prediction_error;
    }
  }

  std::vector<float> updated_w = w;
  std::vector<float> updated_b = b;
  for (size_t class_index = 0; class_index < num_classes; class_index++) {
    for (size_t j = 0; j < d; j++) {
      updated_w[class_index * d + j] -=
          alpha *
          (((1.0f / m) * dw[class_index * d + j]) +
           (l2_regularisation_strength * w[class_index * d + j]));
    }
    updated_b[class_index] -= alpha * ((1.0f / m) * db[class_index]);
  }

  return {updated_w, updated_b};
}

inline float avg_loss(const std::vector<std::vector<float>> &X,
                      const std::vector<float> &y,
                      const std::vector<float> &w,
                      const std::vector<float> &b,
                      float l2_regularisation_strength) {
  float m = X.size();
  float total_loss = 0;
  float epsilon = 1e-7;

  for (size_t i = 0; i < m; i++) {
    std::vector<float> hat_y = predict(X[i], w, b);
    int true_class_index = static_cast<int>(y[i]);
    float predicted_probability = hat_y[true_class_index];
    predicted_probability =
        std::max(epsilon, std::min(1.0f - epsilon, predicted_probability));
    total_loss += -log(predicted_probability);
  }

  float weights_squared_sum = 0;
  for (float weight_value : w) {
    weights_squared_sum += weight_value * weight_value;
  }
  float l2_penalty_term = 0.5f * l2_regularisation_strength * weights_squared_sum;

  float J = total_loss / m;
  return J + l2_penalty_term;
}

inline SoftmaxRegressionResult
train(const std::vector<std::vector<float>> &X,
      const std::vector<float> &y,
      const std::vector<float> &initial_w,
      const std::vector<float> &initial_b, float alpha,
      float l2_regularisation_strength, int num_epochs) {
  std::vector<float> current_w = initial_w;
  std::vector<float> current_b = initial_b;
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    SoftmaxRegressionResult result = update_w_and_b(
        X, y, current_w, current_b, alpha, l2_regularisation_strength);
    current_w = result.w;
    current_b = result.b;

    if (epoch % 1000 == 0) {
      float loss = avg_loss(X, y, current_w, current_b, l2_regularisation_strength);
      printf("Epoch %d - Current Log Loss: %f\n", epoch, loss);
    }
  }
  printf("Training complete. Final Log Loss: %f\n",
         avg_loss(X, y, current_w, current_b, l2_regularisation_strength));
  return {current_w, current_b};
}
