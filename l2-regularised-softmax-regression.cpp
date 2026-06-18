#include "l2-regularised-softmax-regression.hpp"
#include "standardisation.hpp"
#include <algorithm>
#include <vector>

int main() {
  std::vector<std::vector<float>> features = {{10, 5}, {20, 6}, {30, 7},
                                              {40, 8}, {50, 9}, {60, 10}};
  std::vector<float> classes = {0, 0, 1, 1, 2, 2};

  auto [x_scaled, x_means, x_std_devs] = standardise_matrix(features);

  std::vector<std::vector<float>> w_init = {{0, 0}, {0, 0}, {0, 0}};
  std::vector<float> b_init = {0, 0, 0};

  auto [w, b] = train(x_scaled, classes, w_init, b_init, 0.1, 0.01, 10000);

  std::vector<float> x_new = {55, 9.5f};
  std::vector<float> probabilities =
      predict(standardise_vector(x_new, x_means, x_std_devs), w, b);

  printf("Prediction for x = %f, %f:\n", x_new[0], x_new[1]);
  for (size_t i = 0; i < probabilities.size(); i++) {
    printf("  Probability of class %zu = %f\n", i, probabilities[i]);
  }

  auto max_it = std::max_element(probabilities.begin(), probabilities.end());
  int decision = std::distance(probabilities.begin(), max_it);
  printf("Classification Decision: Class %d\n", decision);
}
