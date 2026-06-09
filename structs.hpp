#pragma once
#include <cstddef>
#include <vector>

namespace ml {
struct feature_vector {
  std::vector<float> v;

  feature_vector(size_t dimensions) : v(dimensions, 0) {}

  float &operator[](size_t i) { return v[i]; }
  const float &operator[](size_t i) const { return v[i]; }

  size_t size() const { return v.size(); }
};

struct linear_model {
  feature_vector weights;
  float bias;
};
} // namespace ml
