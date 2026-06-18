#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

inline std::vector<std::vector<int>>
encode(const std::vector<char> &input_data, const std::vector<char> &categories) {
  int num_categories = categories.size();

  std::vector<std::vector<int>> encoded_matrix;
  encoded_matrix.reserve(input_data.size());

  for (char data_item : input_data) {
    auto category_iterator = std::find(categories.begin(), categories.end(), data_item);

    std::vector<int> one_hot_vector(num_categories, 0);

    if (category_iterator != categories.end()) {
      int category_index = std::distance(categories.begin(), category_iterator);
      one_hot_vector[category_index] = 1;
    }

    encoded_matrix.push_back(one_hot_vector);
  }

  return encoded_matrix;
}
