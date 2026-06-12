#include "support-vector-machine.hpp"

int main() {
  std::vector<float> hours_studied = {1, 2, 3, 4, 5, 6};
  std::vector<float> pass_fail = {-1, -1, -1, 1, 1, 1};

  std::vector<float> w_and_b =
      train(hours_studied, pass_fail, 0, 0, 0.05, 0.01, 5000);

  float w = w_and_b[0];
  float b = w_and_b[1];

  printf("Trained model parameters: w = %f, b = %f\n", w, b);
  printf("Decision Boundary is at x = %f\n\n", -b / w);

  float x_new = 3.5;

  float predicted_class = predict(x_new, w, b);

  printf("Prediction for x = %f: Predicted Label = %f\n", x_new,
         predicted_class);

  char decision = (predicted_class == 1) ? 'P' : 'F';
  printf("Classification Decision: Class %c\n", decision);
}
