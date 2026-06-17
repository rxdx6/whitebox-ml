#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct SoftmaxRegressionResult {
  std::vector<float> w;
  std::vector<float> b;
};

inline std::vector<float> predict(float x, const std::vector<float> &w,
                                  const std::vector<float> &b) {
  size_t K = w.size();
  std::vector<float> logits(K);
  float max_logit = -INFINITY;
  for (size_t k = 0; k < K; k++) {
    logits[k] = w[k] * x + b[k];
    if (logits[k] > max_logit) {
      max_logit = logits[k];
    }
  }
  std::vector<float> probs(K);
  float sum = 0;
  for (size_t k = 0; k < K; k++) {
    probs[k] = exp(logits[k] - max_logit);
    sum += probs[k];
  }
  for (size_t k = 0; k < K; k++) {
    probs[k] /= sum;
  }
  return probs;
}

inline SoftmaxRegressionResult update_w_and_b(const std::vector<float> &X,
                                              const std::vector<float> &y,
                                              const std::vector<float> &w,
                                              const std::vector<float> &b,
                                              float alpha, float lambda) {
  size_t K = w.size();
  float N = X.size();
  std::vector<float> dl_dw(K, 0.0f);
  std::vector<float> dl_db(K, 0.0f);

  for (size_t i = 0; i < N; i++) {
    std::vector<float> probs = predict(X[i], w, b);
    int target_class = static_cast<int>(y[i]);
    for (size_t k = 0; k < K; k++) {
      float indicator = (target_class == static_cast<int>(k)) ? 1.0f : 0.0f;
      float diff = probs[k] - indicator;
      dl_dw[k] += diff * X[i];
      dl_db[k] += diff;
    }
  }

  std::vector<float> new_w = w;
  std::vector<float> new_b = b;
  for (size_t k = 0; k < K; k++) {
    new_w[k] -= alpha * (((1.0f / N) * dl_dw[k]) + (lambda * w[k]));
    new_b[k] -= alpha * ((1.0f / N) * dl_db[k]);
  }

  return {new_w, new_b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      const std::vector<float> &w, const std::vector<float> &b,
                      float lambda) {
  float N = X.size();
  float total_loss = 0;
  float epsilon = 1e-7;

  for (size_t i = 0; i < N; i++) {
    std::vector<float> probs = predict(X[i], w, b);
    int target_class = static_cast<int>(y[i]);
    float pred = probs[target_class];
    pred = std::max(epsilon, std::min(1.0f - epsilon, pred));
    total_loss += -log(pred);
  }

  float w_squared_sum = 0;
  for (float val : w) {
    w_squared_sum += val * val;
  }
  float l2_penalty = 0.5f * lambda * w_squared_sum;

  return (total_loss / N) + l2_penalty;
}

inline SoftmaxRegressionResult train(const std::vector<float> &X,
                                     const std::vector<float> &y,
                                     const std::vector<float> &w,
                                     const std::vector<float> &b, float alpha,
                                     float lambda, int epochs) {
  std::vector<float> current_w = w;
  std::vector<float> current_b = b;
  for (int e = 0; e < epochs; e++) {
    SoftmaxRegressionResult result =
        update_w_and_b(X, y, current_w, current_b, alpha, lambda);
    current_w = result.w;
    current_b = result.b;

    if (e % 1000 == 0) {
      float current_loss = avg_loss(X, y, current_w, current_b, lambda);
      printf("Epoch %d - Current Log Loss: %f\n", e, current_loss);
    }
  }
  printf("Training complete. Final Log Loss: %f\n",
         avg_loss(X, y, current_w, current_b, lambda));
  return {current_w, current_b};
}
