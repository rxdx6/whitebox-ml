#pragma once

#include "l2-regularised-linear-regression.hpp"
#include <cstddef>
#include <vector>

struct NeuralLayer {
  std::vector<LinearRegressionResult> neurons;
};

inline float relu(float x) { return x > 0 ? x : 0; }

inline std::vector<std::vector<float>>
forward_layer(const std::vector<std::vector<float>> &input_batch,
              const NeuralLayer &layer, bool apply_activation = true) {
  size_t num_samples = input_batch.size();
  size_t num_neurons = layer.neurons.size();

  std::vector<std::vector<float>> output_batch(
      num_samples, std::vector<float>(num_neurons, 0));

  for (size_t i = 0; i < num_samples; i++) {
    for (size_t n = 0; n < num_neurons; n++) {
      float z = layer.neurons[n].b;
      for (size_t j = 0; j < input_batch[i].size(); j++) {
        z += layer.neurons[n].w[j] * input_batch[i][j];
      }
      output_batch[i][n] = apply_activation ? relu(z) : z;
    }
  }
  return output_batch;
}

inline std::vector<std::vector<float>>
forward_network(const std::vector<std::vector<float>> &X,
                const std::vector<NeuralLayer> &network) {
  std::vector<std::vector<float>> current_input = X;
  for (size_t l = 0; l < network.size(); l++) {
    bool is_last_layer = (l == network.size() - 1);
    current_input = forward_layer(current_input, network[l], !is_last_layer);
  }
  return current_input;
}
