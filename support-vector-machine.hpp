// Decision boundry is given by wx - b = 0
// where w is a real valued vector,
// where b is a real number,
// where x is our input feature vector,
// where wx = w[1]x[1] + w[2]x[2] + ... + w[D]x[D]
// and D is the number of dimensions of the feature vector x
//
// The predict label y is given by y = sign(wx - b)
// where sign is a mathematical operator that takes any value as input and
// returns +1 if the input is a positive number or -1 if the input is a negative
// number
//
// The goal of SVM is to find w* and b* which are optimal w and b such that
// f(x) = sign(w*x - b*)
//
// wx[i] - b >= 1 if y[i] = +1 and
// wx[i] - b <= 1 if y[i] = -1
//
// We would also prefer that the hyperplane separates positive examples from
// negative ones with the largest margin. The margin is the distance between the
// closest examples of two classes, as defined by the decision boundary. A large
// margin contributes to a better generalization, that is how well the model
// will classify new examples in the future. To achieve that, we need to
// minimize the Euclidean norm of w denoted by ||w|| and given by
// sqrt(
// summation j = 1 to D(
// w[j] * w[j];
// )
// )
//
// SVM: Minimize ||w|| subject to y[i](wx[i] - b) >= 1 for i = 1, ..., N. given
// y[i](w * x[i] - b) >= 1

#pragma once

#include <vector>
inline std::vector<float> update_w_and_b(const std::vector<float> &X,
                                         const std::vector<float> &y, float w,
                                         float b, float alpha,
                                         float lambda_param) {
  float dl_dw = 0;
  float dl_db = 0;
  float N = X.size();

  for (size_t i = 0; i < N; i++) {
    if (y[i] * ((w * X[i]) + b) < 1) {
      dl_dw += -y[i] * X[i];
      dl_db += -y[i];
    }
  }

  w -= alpha * (lambda_param * w + (dl_dw / N));
  b -= alpha * (dl_db / N);

  return {w, b};
}

inline float avg_loss(const std::vector<float> &X, const std::vector<float> &y,
                      float w, float b, float lambda_param) {
  float N = X.size();
  float total_hinge_loss = 0;

  for (size_t i = 0; i < N; i++) {
    float margin_distance = 1 - y[i] * ((w * X[i]) + b);
    if (margin_distance > 0) {
      total_hinge_loss += margin_distance;
    }
  }

  float regular_loss = 0.5 * lambda_param * w * w;
  return regular_loss + (total_hinge_loss / N);
}

inline std::vector<float> train(const std::vector<float> &X,
                                const std::vector<float> &y, float w, float b,
                                float alpha, float lambda_param, int epochs) {
  for (int e = 0; e < epochs; e++) {
    std::vector<float> w_and_b =
        update_w_and_b(X, y, w, b, alpha, lambda_param);
    w = w_and_b[0];
    b = w_and_b[1];
  }

  return {w, b};
}

inline float predict(float x, float w, float b) {
  float raw_score = (w * x) + b;
  return (raw_score >= 0) ? 1 : -1;
}
