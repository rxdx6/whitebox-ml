#pragma once

#include <vector>
inline std::vector<float> update_w_and_b(const std::vector<float> &X,
                                         const std::vector<float> &y, float w,
                                         float b, float alpha) {
  float dw = 0;
  float db = 0;
  float m = X.size();

  for (size_t i = 0; i < m; i++) {
    dw += -2 * X[i] * (y[i] - (w * X[i] + b));
    db += -2 * (y[i] - (w * X[i] + b));
  }

  w -= (1 / m) * dw * alpha;
  b -= (1 / m) * db * alpha;

  return {w, b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      float w, float b) {
  float m = X.size();
  float J = 0;

  for (size_t i = 0; i < m; i++) {
    J += (y[i] - (w * X[i] + b)) * (y[i] - (w * X[i] + b));
  }

  return J / m;
}

inline std::vector<float> train(const std::vector<float> &X,
                                const std::vector<float> &y, float w, float b,
                                float alpha, int num_epochs) {
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<float> updated_w_and_b = update_w_and_b(X, y, w, b, alpha);
    w = updated_w_and_b[0];
    b = updated_w_and_b[1];
  }

  return {w, b};
}

inline float predict(float x, float w, float b) { return (w * x) + b; }
