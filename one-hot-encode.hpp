#pragma once

#include <algorithm>
#include <iterator>
#include <vector>

inline std::vector<std::vector<int>>
encode(const std::vector<char> &data, const std::vector<char> &possibilities) {
  int number_of_categories = possibilities.size();

  std::vector<std::vector<int>> result;
  result.reserve(data.size());

  for (char item : data) {
    auto it = std::find(possibilities.begin(), possibilities.end(), item);

    std::vector<int> one_hot(number_of_categories, 0);

    if (it != possibilities.end()) {
      int index = std::distance(possibilities.begin(), it);
      one_hot[index] = 1;
    }

    result.push_back(one_hot);
  }

  return result;
}
