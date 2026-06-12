#include "closed-form-linear-regression.hpp"
#include <iostream>
#include <vector>

int main() {
  std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {3.0f}, {4.0f}};

  std::vector<float> y = {3.0f, 5.0f, 7.0f, 9.0f};

  std::cout << "--- Linear Regression Demo ---" << std::endl;
  std::cout << "Training model on 4 data points..." << std::endl;

  try {
    std::vector<float> weights = linear_regressor(X, y);

    std::cout << "\nCalculated Weights:" << std::endl;
    std::cout << "Slope (w0):       " << weights[0] << " (Expected: ~2.0)"
              << std::endl;
    std::cout << "Intercept (bias): " << weights[1] << " (Expected: ~1.0)"
              << std::endl;

    float new_x = 5.0f;
    float predicted_y = (weights[0] * new_x) + weights[1];

    std::cout << "\nTesting prediction:" << std::endl;
    std::cout << "For x = " << new_x << ", predicted y = " << predicted_y
              << " (Expected: 11.0)" << std::endl;

  } catch (const std::invalid_argument &e) {
    std::cerr << "Input Error: " << e.what() << std::endl;
    return 1;
  } catch (const std::runtime_error &e) {
    std::cerr << "Runtime Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
