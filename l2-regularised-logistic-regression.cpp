#include "l2-regularised-logistic-regression.hpp"
#include "standardisation.hpp"
#include <vector>

int main() {
  std::vector<float> hours_studied = {1, 2, 3, 4, 5, 6};
  std::vector<float> pass_fail = {0, 0, 0, 1, 1, 1};

  auto [x_scaled, x_mean, x_standard_deviation] = standardise(hours_studied);

  std::vector<float> w_and_b =
      train(x_scaled, pass_fail, 0, 0, 0.1, 0.01, 10000);

  float w = w_and_b[0];
  float b = w_and_b[1];

  float x_new = 5.5;
  float probability =
      predict(standardise(x_new, x_mean, x_standard_deviation), w, b);

  printf("Prediction for x = %f: Probability of passing = %f\n", x_new,
         probability);

  char decision = (probability >= 0.5) ? 'P' : 'F';
  printf("Classification Decision (Threshold 0.5): Class %c\n", decision);
}
