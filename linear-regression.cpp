#include "linear-regression.hpp"

int main() {
  std::vector<float> X = {37.8, 39.3, 45.9, 41.3};
  std::vector<float> y = {22.1, 10.4, 9.3, 18.5};

  std::vector<float> w_and_b = train(X, y, 0, 0, 0.0001, 15000);

  float x_new = 23;
  float y_new = predict(x_new, w_and_b[0], w_and_b[1]);
  printf("Prediction for x = %f: y = %f\n", x_new, y_new);
}
