// f(x) = wx + b
// Objective function (minimise this): 1/N(summation((f(x[i]) - y[i])^2))

#include "structs.hpp"
#include <vector>

ml::linear_model linear_regressor(const std::vector<ml::feature_vector> &X,
                                  const std::vector<float> &y) {
  ml::feature_vector w{X[0].size()};
  float b = 0;

  // optimise

  return {w, b};
}
