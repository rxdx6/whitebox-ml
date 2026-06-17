#include "l2-regularised-softmax-regression.hpp"
#include "standardisation.hpp"
#include <algorithm>
#include <vector>

int main() {
  std::vector<float> scores = {10, 20, 30, 40, 50, 60};
  std::vector<float> classes = {0, 0, 1, 1, 2, 2};

  auto [x_scaled, x_mean, x_standard_deviation] = standardise(scores);

  std::vector<float> w_init = {0, 0, 0};
  std::vector<float> b_init = {0, 0, 0};

  auto [w, b] = train(x_scaled, classes, w_init, b_init, 0.1, 0.01, 10000);

  float x_new = 55;
  std::vector<float> probabilities =
      predict(standardise(x_new, x_mean, x_standard_deviation), w, b);

  printf("Prediction for x = %f:\n", x_new);
  for (size_t i = 0; i < probabilities.size(); i++) {
    printf("  Probability of class %zu = %f\n", i, probabilities[i]);
  }

  auto max_it = std::max_element(probabilities.begin(), probabilities.end());
  int decision = std::distance(probabilities.begin(), max_it);
  printf("Classification Decision: Class %d\n", decision);
}
