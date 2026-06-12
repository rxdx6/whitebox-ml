#pragma once

#include <algorithm>
#include <vector>

inline std::vector<float> normaliser(std::vector<float> x) {
  if (x.empty()) {
    return {};
  }

  auto [min_it, max_it] = std::minmax_element(x.begin(), x.end());
  float minj = *min_it;
  float maxj = *max_it;
  float range = maxj - minj;

  if (range == 0) {
    std::fill(x.begin(), x.end(), 0);
    return x;
  }

  for (float &val : x) {
    val = (val - minj) / range;
  }

  return x;
}
