#pragma once

#include <vector>
inline std::vector<float> update_w_and_b(const std::vector<float> &X,
                                         const std::vector<float> &y, float w,
                                         float b, float alpha,
                                         float l2_regularisation_strength) {
  float dw = 0;
  float db = 0;
  float m = X.size();

  for (size_t i = 0; i < m; i++) {
    if (y[i] * ((w * X[i]) + b) < 1) {
      dw += -y[i] * X[i];
      db += -y[i];
    }
  }

  w -= alpha * (l2_regularisation_strength * w + (dw / m));
  b -= alpha * (db / m);

  return {w, b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      float w, float b, float l2_regularisation_strength) {
  float m = X.size();
  float hinge_loss_sum = 0;

  for (size_t i = 0; i < m; i++) {
    float margin_distance = 1 - y[i] * ((w * X[i]) + b);
    if (margin_distance > 0) {
      hinge_loss_sum += margin_distance;
    }
  }

  float l2_penalty_term = 0.5 * l2_regularisation_strength * w * w;
  float J = hinge_loss_sum / m;
  return l2_penalty_term + J;
}

inline std::vector<float> train(const std::vector<float> &X,
                                const std::vector<float> &y, float w, float b,
                                float alpha, float l2_regularisation_strength, int num_epochs) {
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<float> updated_w_and_b =
        update_w_and_b(X, y, w, b, alpha, l2_regularisation_strength);
    w = updated_w_and_b[0];
    b = updated_w_and_b[1];
  }

  return {w, b};
}

inline float predict(float x, float w, float b) {
  float hat_y = (w * x) + b;
  return (hat_y >= 0) ? 1 : -1;
}
