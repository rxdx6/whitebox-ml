#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

inline std::vector<float> standardise(const std::vector<float> &v) {
  if (v.empty()) {
    return {};
  }

  float sum = std::accumulate(v.begin(), v.end(), 0.0f);
  float average = sum / v.size();

  float variance = 0;
  for (const float val : v) {
    float diff = val - average;
    variance += diff * diff;
  }
  float standard_deviation = std::sqrt(variance / v.size());
  if (standard_deviation == 0.0f) {
    return std::vector<float>(v.size(), 0.0f);
  }

  std::vector<float> res;
  res.reserve(v.size());
  for (const float val : v) {
    res.push_back((val - average) / standard_deviation);
  }
  return res;
}
