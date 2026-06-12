#pragma once

#include <vector>
inline std::vector<float> update_w_and_b(const std::vector<float> &X,
                                         const std::vector<float> &y, float w,
                                         float b, float alpha) {
  float dl_dw = 0;
  float dl_db = 0;
  float N = X.size();

  for (size_t i = 0; i < N; i++) {
    dl_dw += -2 * X[i] * (y[i] - (w * X[i] + b));
    dl_db += -2 * (y[i] - (w * X[i] + b));
  }

  w -= (1 / N) * dl_dw * alpha;
  b -= (1 / N) * dl_db * alpha;

  return {w, b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      float w, float b) {
  float N = X.size();
  float total_error = 0;

  for (size_t i = 0; i < N; i++) {
    total_error += (y[i] - (w * X[i] + b)) * (y[i] - (w * X[i] + b));
  }

  return total_error / N;
}

inline std::vector<float> train(const std::vector<float> &X,
                                const std::vector<float> &y, float w, float b,
                                float alpha, int epochs) {
  for (int e = 0; e < epochs; e++) {
    std::vector<float> w_and_b = update_w_and_b(X, y, w, b, alpha);
    w = w_and_b[0];
    b = w_and_b[1];
  }

  return {w, b};
}

inline float predict(float x, float w, float b) { return (w * x) + b; }
