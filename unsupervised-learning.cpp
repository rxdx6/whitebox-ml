#include "feature-vector.hpp"
#include <cstddef>
#include <vector>

struct dataset {
  std::vector<ml::feature_vector> x;
  size_t number_of_datapoints() const { return x.size(); }
};

int main() { return 0; }
