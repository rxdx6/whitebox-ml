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
  float dl_dw = 0;
  float dl_db = 0;
  float N = X.size();

  for (size_t i = 0; i < N; i++) {
    float pred = predict(X[i], w, b);
    dl_dw += (pred - y[i]) * X[i];
    dl_db += pred - y[i];
  }

  w -= (1 / N) * dl_dw * alpha;
  b -= (1 / N) * dl_db * alpha;

  return {w, b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      float w, float b) {
  float N = X.size();
  float total_loss = 0;
  float epsilon = 1e-7;

  for (size_t i = 0; i < N; i++) {
    float pred = predict(X[i], w, b);
    pred = std::max(epsilon, std::min(1 - epsilon, pred));
    total_loss += -(y[i] * log(pred) + (1 - y[i]) * log(1 - pred));
  }

  return total_loss / N;
}

inline std::vector<float> train(const std::vector<float> &X,
                                const std::vector<float> &y, float w, float b,
                                float alpha, int epochs) {
  for (int e = 0; e < epochs; e++) {
    std::vector<float> w_and_b = update_w_and_b(X, y, w, b, alpha);
    w = w_and_b[0];
    b = w_and_b[1];

    if (e % 1000 == 0) {
      float current_loss = avg_loss(X, y, w, b);
      printf("Epoch %d - Current Log Loss: %f\n", e, current_loss);
    }
  }
  printf("Training complete. Final Log Loss: %f\n", avg_loss(X, y, w, b));
  return {w, b};
}
