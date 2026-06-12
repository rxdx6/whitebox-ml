// f(x) = 1 / (1 + e^(-(wx + b)))
// To optimise: Maximise: (Products of i = 1 to N)(f(x[i])^y[i] * (1 -
// f(x[i]))^(1 - y[i]))
//
// It is more convenient to maximise the log likelihood instead of likelihood
// due to the exponential nature of f(x). The log likelihood is given by:
// Summation(y[i]ln(f(x[i])) + (1 - y[i])ln(1 - f(x[i])))

#include "logistic-regression.hpp"
#include <vector>

int main() {
  std::vector<float> hours_studied = {1, 2, 3, 4, 5, 6};
  std::vector<float> pass_fail = {0, 0, 0, 1, 1, 1};

  std::vector<float> w_and_b =
      train(hours_studied, pass_fail, 0, 0, 0.1, 10000);

  float w = w_and_b[0];
  float b = w_and_b[1];

  float x_new = 3.5;
  float probability = predict(x_new, w, b);

  printf("Prediction for x = %f: Probability of passing = %f\n", x_new,
         probability);

  char decision = (probability >= 0.5) ? 'P' : 'F';
  printf("Classification Decision (Threshold 0.5): Class %c\n", decision);
}
