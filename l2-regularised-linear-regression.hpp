#pragma once

#include <vector>

struct LinearRegressionResult {
  std::vector<float> w;
  float b;
};

inline LinearRegressionResult
update_w_and_b(const std::vector<std::vector<float>> &X,
               const std::vector<float> &y, const std::vector<float> &w,
               float b, float alpha, float lambda) {
  float dl_db = 0.0f;
  float N = X.size();
  size_t num_features = w.size();
  std::vector<float> dl_dw(num_features, 0.0f);

  for (size_t i = 0; i < N; i++) {
    float pred = b;
    for (size_t j = 0; j < num_features; j++) {
      pred += w[j] * X[i][j];
    }

    float diff = y[i] - pred;
    for (size_t j = 0; j < num_features; j++) {
      dl_dw[j] += -2 * X[i][j] * diff;
    }
    dl_db += -2 * diff;
  }

  std::vector<float> new_w = w;
  for (size_t j = 0; j < num_features; j++) {
    new_w[j] -= alpha * (((1.0f / N) * dl_dw[j]) + (lambda * w[j]));
  }
  float new_b = b - (1.0f / N) * dl_db * alpha;

  return {new_w, new_b};
}

inline float avg_loss(const std::vector<std::vector<float>> &X,
                      const std::vector<float> &y, const std::vector<float> &w,
                      float b, float lambda) {
  float N = X.size();
  float total_error = 0.0f;
  size_t num_features = w.size();

  for (size_t i = 0; i < N; i++) {
    float pred = b;
    for (size_t j = 0; j < num_features; j++) {
      pred += w[j] * X[i][j];
    }
    float error = y[i] - pred;
    total_error += error * error;
  }

  float mse = total_error / N;
  float w_squared_sum = 0.0f;
  for (float val : w) {
    w_squared_sum += val * val;
  }
  float l2_penalty = 0.5f * lambda * w_squared_sum;

  return mse + l2_penalty;
}

inline LinearRegressionResult train(const std::vector<std::vector<float>> &X,
                                    const std::vector<float> &y,
                                    const std::vector<float> &w, float b,
                                    float alpha, float lambda, int epochs) {
  std::vector<float> current_w = w;
  float current_b = b;
  for (int e = 0; e < epochs; e++) {
    LinearRegressionResult w_and_b =
        update_w_and_b(X, y, current_w, current_b, alpha, lambda);
    current_w = w_and_b.w;
    current_b = w_and_b.b;
  }

  return {current_w, current_b};
}

inline std::vector<float> predict(const std::vector<std::vector<float>> &X,
                                  const std::vector<float> &w, float b) {
  std::vector<float> preds(X.size());
  for (size_t i = 0; i < X.size(); i++) {
    float pred = b;
    for (size_t j = 0; j < w.size(); j++) {
      pred += w[j] * X[i][j];
    }
    preds[i] = pred;
  }
  return preds;
}
