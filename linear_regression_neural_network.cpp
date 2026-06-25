#include "linear_regression_neural_network.hpp"
#include "standardisation.hpp"
#include <cstdio>
#include <vector>

int main() {
  std::vector<std::vector<float>> x = {{1.0f, 2.0f}, {2.0f, 3.0f}, {3.0f, 1.0f},
                                       {4.0f, 3.0f}, {5.0f, 2.0f}, {6.0f, 5.0f},
                                       {7.0f, 2.0f}, {8.0f, 4.0f}};
  std::vector<std::vector<float>> y = {{0.5f}, {1.5f}, {5.5f},  {5.5f},
                                       {8.5f}, {7.5f}, {12.5f}, {12.5f}};

  auto [x_scaled, x_means, x_std_devs] = standardise_matrix(x);

  std::vector<NetworkLayer> network =
      train(x_scaled, y, 3, 128, 0.005f, 0.001f, 100000, true);

  std::vector<std::vector<float>> x_new = {{5.0f, 2.0f}, {3.5f, 1.5f}};
  std::vector<std::vector<float>> x_new_scaled;
  for (const auto &v : x_new) {
    x_new_scaled.push_back(standardise_vector(v, x_means, x_std_devs));
  }

  std::vector<std::vector<float>> predictions = predict(network, x_new_scaled);

  for (size_t i = 0; i < x_new.size(); ++i) {
    printf("Prediction for x = {%f, %f}: y = %f\n", x_new[i][0], x_new[i][1],
           predictions[i][0]);
  }
}
