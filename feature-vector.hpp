#pragma once
#include <vector>

namespace ml {
struct feature_vector {
  std::vector<float> v;
  feature_vector(size_t dimensions = 0) : v(dimensions, 0) {}
};
} // namespace ml
