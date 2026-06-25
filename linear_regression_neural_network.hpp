#pragma once

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

struct NetworkLayer {
  size_t num_inputs = 0;
  size_t num_neurons = 0;

  std::vector<float> weights;
  std::vector<float> biases;

  std::vector<float> dw;
  std::vector<float> db;

  std::vector<float> last_input;
  std::vector<float> last_z;
  std::vector<float> last_output;
  size_t last_m = 0;
};

inline float relu(float x) { return x > 0 ? x : 0; }

inline float relu_derivative(float z) { return z > 0 ? 1 : 0; }

inline std::vector<std::vector<float>>
forward_layer(NetworkLayer &layer,
              const std::vector<std::vector<float>> &input_batch,
              bool apply_activation) {
  size_t m = input_batch.size();
  size_t d = layer.num_inputs;
  size_t n = layer.num_neurons;

  layer.last_m = m;
  layer.last_input.resize(m * d);
  for (size_t i = 0; i < m; i++) {
    std::copy(input_batch[i].begin(), input_batch[i].end(),
              layer.last_input.begin() + i * d);
  }

  layer.last_z.resize(m * n);
  layer.last_output.resize(m * n);

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
              (int)m, (int)n, (int)d, 1.0f,
              layer.last_input.data(), (int)d,
              layer.weights.data(), (int)d,
              0.0f, layer.last_z.data(), (int)n);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      float z = layer.last_z[i * n + j] + layer.biases[j];
      layer.last_z[i * n + j] = z;
      layer.last_output[i * n + j] = apply_activation ? relu(z) : z;
    }
  }

  std::vector<std::vector<float>> result(m, std::vector<float>(n));
  for (size_t i = 0; i < m; i++) {
    std::copy(layer.last_output.begin() + i * n,
              layer.last_output.begin() + (i + 1) * n,
              result[i].begin());
  }
  return result;
}

inline std::vector<std::vector<float>>
forward_network(std::vector<NetworkLayer> &network,
                const std::vector<std::vector<float>> &X) {
  std::vector<std::vector<float>> current_features = X;

  for (size_t l = 0; l < network.size(); l++) {
    bool is_last_layer = (l == network.size() - 1);
    current_features =
        forward_layer(network[l], current_features, !is_last_layer);
  }
  return current_features;
}

inline void backward_network(std::vector<NetworkLayer> &network,
                             const std::vector<std::vector<float>> &y_true) {
  size_t num_layers = network.size();
  size_t m = y_true.size();

  NetworkLayer &output_layer = network[num_layers - 1];
  size_t n_out = output_layer.num_neurons;

  std::vector<float> delta(m * n_out);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n_out; j++) {
      float y_pred = output_layer.last_output[i * n_out + j];
      delta[i * n_out + j] = (2.0f / m) * (y_pred - y_true[i][j]);
    }
  }

  for (int l = static_cast<int>(num_layers) - 1; l >= 0; l--) {
    NetworkLayer &layer = network[l];
    size_t n = layer.num_neurons;
    size_t d = layer.num_inputs;

    layer.dw.assign(n * d, 0.0f);
    layer.db.assign(n, 0.0f);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                (int)n, (int)d, (int)m, 1.0f,
                delta.data(), (int)n,
                layer.last_input.data(), (int)d,
                0.0f, layer.dw.data(), (int)d);

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
        layer.db[j] += delta[i * n + j];
      }
    }

    if (l == 0) break;

    NetworkLayer &prev_layer = network[l - 1];

    std::vector<float> error(m * d);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)m, (int)d, (int)n, 1.0f,
                delta.data(), (int)n,
                layer.weights.data(), (int)d,
                0.0f, error.data(), (int)d);

    std::vector<float> next_delta(m * d);
    for (size_t i = 0; i < m * d; i++) {
      next_delta[i] = error[i] * relu_derivative(prev_layer.last_z[i]);
    }

    delta = std::move(next_delta);
  }
}

inline void update_network_weights(std::vector<NetworkLayer> &network,
                                   float alpha,
                                   float l2_regularisation_strength) {
  for (auto &layer : network) {
    for (size_t i = 0; i < layer.weights.size(); i++) {
      layer.weights[i] -=
          alpha * (layer.dw[i] + l2_regularisation_strength * layer.weights[i]);
    }
    for (size_t i = 0; i < layer.biases.size(); i++) {
      layer.biases[i] -= alpha * layer.db[i];
    }
  }
}

inline float
calculate_network_loss(const std::vector<NetworkLayer> &network,
                       const std::vector<std::vector<float>> &y_pred,
                       const std::vector<std::vector<float>> &y_true,
                       float l2_regularisation_strength) {
  size_t m = y_true.size();
  size_t num_outputs = y_true[0].size();
  float total_mse = 0;

  for (size_t i = 0; i < m; i++) {
    for (size_t n = 0; n < num_outputs; n++) {
      float diff = y_pred[i][n] - y_true[i][n];
      total_mse += diff * diff;
    }
  }
  float average_mse = total_mse / m;

  float weights_squared_sum = 0;
  for (const auto &layer : network) {
    for (float w_val : layer.weights) {
      weights_squared_sum += w_val * w_val;
    }
  }
  float l2_penalty = 0.5f * l2_regularisation_strength * weights_squared_sum;

  return average_mse + l2_penalty;
}

inline void train_network(std::vector<NetworkLayer> &network,
                          const std::vector<std::vector<float>> &X,
                          const std::vector<std::vector<float>> &y, float alpha,
                          float l2_regularisation_strength, int num_epochs,
                          bool print_progress = true) {
  for (int epoch = 0; epoch < num_epochs; epoch++) {
    std::vector<std::vector<float>> predictions = forward_network(network, X);

    backward_network(network, y);

    update_network_weights(network, alpha, l2_regularisation_strength);

    if (print_progress &&
        (epoch == 0 || (epoch + 1) % 100 == 0 || epoch == num_epochs - 1)) {
      float loss = calculate_network_loss(network, predictions, y,
                                          l2_regularisation_strength);
      std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs
                << " -> Total Loss: " << loss << "\n";
    }
  }
}

inline std::vector<NetworkLayer>
train(const std::vector<std::vector<float>> &X,
      const std::vector<std::vector<float>> &y, int num_layers, int num_neurons,
      float alpha, float l2_regularisation_strength, int num_epochs,
      bool print_progress = true) {
  size_t num_features = X[0].size();
  size_t num_outputs = y[0].size();
  std::vector<NetworkLayer> network(num_layers);

  std::mt19937 gen(42);

  for (int l = 0; l < num_layers; l++) {
    bool is_last_layer = (l == num_layers - 1);
    size_t n = is_last_layer ? num_outputs : num_neurons;
    size_t d = (l == 0) ? num_features : num_neurons;

    network[l].num_inputs = d;
    network[l].num_neurons = n;
    network[l].weights.resize(n * d);
    network[l].biases.resize(n, 0.0f);
    network[l].dw.resize(n * d, 0.0f);
    network[l].db.resize(n, 0.0f);

    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / d));

    for (size_t i = 0; i < n * d; i++) {
      network[l].weights[i] = dist(gen);
    }
  }

  train_network(network, X, y, alpha, l2_regularisation_strength, num_epochs,
                print_progress);
  return network;
}

inline std::vector<std::vector<float>>
predict(std::vector<NetworkLayer> &network,
        const std::vector<std::vector<float>> &X) {
  return forward_network(network, X);
}
