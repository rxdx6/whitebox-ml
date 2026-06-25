#include "linear_regression_neural_network.hpp"
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  std::ifstream file("auto-mpg.csv");
  std::string line;
  std::getline(file, line);
  std::vector<std::vector<float>> x;
  std::vector<std::vector<float>> y;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::stringstream ss(line);
    std::string val;
    std::vector<float> row;
    while (std::getline(ss, val, ',')) {
      row.push_back(std::stof(val));
    }
    if (row.empty()) {
      continue;
    }
    y.push_back({row[0]});
    x.push_back(std::vector<float>(row.begin() + 1, row.end()));
  }
  size_t split = x.size() / 2;
  std::vector<std::vector<float>> x_train(x.begin(), x.begin() + split);
  std::vector<std::vector<float>> y_train(y.begin(), y.begin() + split);
  std::vector<std::vector<float>> x_test(x.begin() + split, x.end());
  std::vector<std::vector<float>> y_test(y.begin() + split, y.end());
  std::vector<NetworkLayer> network =
      train(x_train, y_train, 3, 128, 0.005f, 0.001f, 10000, true);
  std::vector<std::vector<float>> predictions = predict(network, x_test);
  float test_loss =
      calculate_network_loss(network, predictions, y_test, 0.001f);
  printf("Test Loss: %f\n", test_loss);
}
