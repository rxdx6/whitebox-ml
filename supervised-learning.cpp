#include "feature-vector.hpp"
#include <vector>

template <typename target_type = float> struct dataset {
  std::vector<ml::feature_vector> x;
  std::vector<target_type> y;
  size_t number_of_datapoints() const { return x.size(); }
};

int main() { return 0; }
