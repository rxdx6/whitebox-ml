#include "softmax_regression_neural_network.hpp"
#include "one-hot-encode.hpp"
#include <cstdio>
#include <vector>

int main() {
  std::vector<std::vector<float>> features = {
      {1.5f, 0.2f}, {-0.5f, 2.1f}, {0.1f, -1.2f}, {2.0f, 1.8f}};

  std::vector<char> input_data = {'A', 'B', 'C', 'A'};
  std::vector<char> categories = {'A', 'B', 'C'};

  auto encoded_int = encode(input_data, categories);
  std::vector<std::vector<float>> classes;
  for (const auto &row : encoded_int) {
    classes.push_back(std::vector<float>(row.begin(), row.end()));
  }

  int num_layers = 3;
  int num_neurons = 128;
  float alpha = 0.005f;
  float l2_regularisation_strength = 0.001f;
  int num_epochs = 10000;

  auto network = train(features, classes, num_layers, num_neurons, alpha,
                       l2_regularisation_strength, num_epochs, true);

  auto predictions = predict(network, features);

  printf("\nFinal Output Probabilities:\n");
  for (size_t i = 0; i < predictions.size(); i++) {
    printf("Input %zu -> [ ", i);
    for (float prob : predictions[i]) {
      printf("%f ", prob);
    }
    printf("]\n");
  }
}
