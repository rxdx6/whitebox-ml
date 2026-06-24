#pragma once

#include <vector>

struct LinearRegressionResult {
  std::vector<float> w;
  float b;
};

inline LinearRegressionResult
update_w_and_b(const std::vector<std::vector<float>> &X,
               const std::vector<float> &y, const std::vector<float> &w,
               float b, float alpha, float l2_regularisation_strength) {
  float db = 0.0f;
  float m = X.size();
  size_t d = w.size();
  std::vector<float> dw(d, 0.0f);

  for (size_t i = 0; i < m; i++) {
    float hat_y = b;
    for (size_t j = 0; j < d; j++) {
      hat_y += w[j] * X[i][j];
    }

    float prediction_error = y[i] - hat_y;
    for (size_t j = 0; j < d; j++) {
      dw[j] += -2 * X[i][j] * prediction_error;
    }
    db += -2 * prediction_error;
  }

  std::vector<float> updated_w = w;
  for (size_t j = 0; j < d; j++) {
    updated_w[j] -=
        alpha * (((1.0f / m) * dw[j]) + (l2_regularisation_strength * w[j]));
  }
  float updated_b = b - (1.0f / m) * db * alpha;

  return {updated_w, updated_b};
}

inline float avg_loss(const std::vector<std::vector<float>> &X,
                      const std::vector<float> &y, const std::vector<float> &w,
                      float b, float l2_regularisation_strength) {
  float m = X.size();
  float total_error = 0.0f;
  size_t d = w.size();

  for (size_t i = 0; i < m; i++) {
    float hat_y = b;
    for (size_t j = 0; j < d; j++) {
      hat_y += w[j] * X[i][j];
    }
    float error = y[i] - hat_y;
    total_error += error * error;
  }

  float J = total_error / m;
  float weights_squared_sum = 0.0f;
  for (float weight_value : w) {
    weights_squared_sum += weight_value * weight_value;
  }
  float l2_penalty_term =
      0.5f * l2_regularisation_strength * weights_squared_sum;

  return J + l2_penalty_term;
}

inline LinearRegressionResult
train(const std::vector<std::vector<float>> &X, const std::vector<float> &y,
      const std::vector<float> &w, float b, float alpha,
      float l2_regularisation_strength, int num_epochs) {
  std::vector<float> current_w = w;
  float current_b = b;
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    LinearRegressionResult result = update_w_and_b(
        X, y, current_w, current_b, alpha, l2_regularisation_strength);
    current_w = result.w;
    current_b = result.b;
  }

  return {current_w, current_b};
}

inline std::vector<float> predict(const std::vector<std::vector<float>> &X,
                                  const std::vector<float> &w, float b) {
  std::vector<float> y_pred(X.size());
  for (size_t i = 0; i < X.size(); i++) {
    float hat_y = b;
    for (size_t j = 0; j < w.size(); j++) {
      hat_y += w[j] * X[i][j];
    }
    y_pred[i] = hat_y;
  }
  return y_pred;
}
