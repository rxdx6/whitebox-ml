#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

inline float sigmoid_function(float z) { return 1 / (1 + exp(-z)); }

inline float predict(float x, float w, float b) {
  return sigmoid_function((w * x) + b);
}

inline std::vector<float> update_w_and_b(const std::vector<float> &X,
                                         const std::vector<float> &y, float w,
                                         float b, float alpha) {
  float dw = 0;
  float db = 0;
  float m = X.size();

  for (size_t i = 0; i < m; i++) {
    float y_pred = predict(X[i], w, b);
    dw += (y_pred - y[i]) * X[i];
    db += y_pred - y[i];
  }

  w -= (1 / m) * dw * alpha;
  b -= (1 / m) * db * alpha;

  return {w, b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      float w, float b) {
  float m = X.size();
  float J = 0;
  float epsilon = 1e-7;

  for (size_t i = 0; i < m; i++) {
    float y_pred = predict(X[i], w, b);
    y_pred = std::max(epsilon, std::min(1 - epsilon, y_pred));
    J += -(y[i] * log(y_pred) + (1 - y[i]) * log(1 - y_pred));
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

    if (epoch % 1000 == 0) {
      float loss = avg_loss(X, y, w, b);
      printf("Epoch %d - Current Log Loss: %f\n", epoch, loss);
    }
  }
  printf("Training complete. Final Log Loss: %f\n", avg_loss(X, y, w, b));
  return {w, b};
}
