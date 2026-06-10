// f(x) = 1 / (1 + e^(-(wx + b)))
// To optimise: Maximise: (Products of i = 1 to N)(f(x[i])^y[i] * (1 -
// f(x[i]))^(1 - y[i]))
//
// It is more convenient to maximise the log likelihood instead of likelihood
// due to the exponential nature of f(x). The log likelihood is given by:
// Summation(y[i]ln(f(x[i])) + (1 - y[i])ln(1 - f(x[i])))

#include "structs.hpp"
#include <vector>

ml::linear_model logistic_regressor(const std::vector<ml::feature_vector> &X,
                                    const std::vector<bool> &y) {
  ml::feature_vector w{X[0].size()};
  float b = 0;

  // optimise

  return {w, b};
}
