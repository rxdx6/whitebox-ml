#include "l2-regularised-linear-regression.hpp"
#include "standardisation.hpp"
#include <cstdio>

int main() {
  std::vector<std::vector<float>> x = {{230.1f, 37.8f, 69.2f},
                                       {44.5f, 39.3f, 45.1f},
                                       {17.2f, 45.9f, 69.3f},
                                       {151.5f, 41.3f, 58.5f}};
  std::vector<float> y = {22.1f, 10.4f, 9.3f, 18.5f};

  auto [x_scaled, x_means, x_standard_deviations] = standardise_matrix(x);
  auto [y_scaled, y_mean, y_standard_deviation] = standardise(y);

  size_t num_features = x[0].size();
  std::vector<float> w_init(num_features, 0.0f);

  auto [w, b] = train(x_scaled, y_scaled, w_init, 0.0f, 0.0001f, 0.01f, 15000);

  std::vector<std::vector<float>> x_new = {{180.8f, 10.8f, 58.4f}};
  std::vector<std::vector<float>> x_new_scaled;
  for (const auto &v : x_new) {
    x_new_scaled.push_back(
        standardise_vector(v, x_means, x_standard_deviations));
  }

  std::vector<float> y_new = predict(x_new_scaled, w, b);
  for (size_t i = 0; i < x_new.size(); ++i) {
    float y_unscaled = unstandardise(y_new[i], y_mean, y_standard_deviation);
    printf("Prediction for x = {%f, %f, %f}: y = %f\n", x_new[i][0],
           x_new[i][1], x_new[i][2], y_unscaled);
  }
}
