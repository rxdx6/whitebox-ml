#pragma once

#include <algorithm>
#include <vector>

inline std::vector<float> normaliser(std::vector<float> values) {
  if (values.empty()) {
    return {};
  }

  auto [min_iterator, max_iterator] = std::minmax_element(values.begin(), values.end());
  float minimum_value = *min_iterator;
  float maximum_value = *max_iterator;
  float value_range = maximum_value - minimum_value;

  if (value_range == 0) {
    std::fill(values.begin(), values.end(), 0);
    return values;
  }

  for (float &value : values) {
    value = (value - minimum_value) / value_range;
  }

  return values;
}
