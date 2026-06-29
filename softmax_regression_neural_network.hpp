#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

struct NetworkNeuron {
  std::vector<float> w;
  float b = 0;

  std::vector<float> dw;
  float db = 0;
};

struct NetworkLayer {
  std::vector<NetworkNeuron> neurons;

  std::vector<std::vector<float>> last_inputs;
  std::vector<std::vector<float>> last_zs;
  std::vector<std::vector<float>> last_outputs;
};

inline float relu(float x) { return x > 0 ? x : 0; }

inline float relu_derivative(float z) { return z > 0 ? 1 : 0; }

inline std::vector<std::vector<float>>
forward_layer(NetworkLayer &layer,
              const std::vector<std::vector<float>> &input_batch,
              bool apply_activation) {
  size_t m = input_batch.size();
  size_t num_neurons = layer.neurons.size();

  layer.last_inputs = input_batch;

  layer.last_zs.assign(m, std::vector<float>(num_neurons, 0));
  layer.last_outputs.assign(m, std::vector<float>(num_neurons, 0));

  for (size_t i = 0; i < m; i++) {
    for (size_t n = 0; n < num_neurons; n++) {
      float z = layer.neurons[n].b;
      for (size_t j = 0; j < input_batch[i].size(); j++) {
        z += layer.neurons[n].w[j] * input_batch[i][j];
      }
      layer.last_zs[i][n] = z;
      layer.last_outputs[i][n] = apply_activation ? relu(z) : z;
    }
  }
  return layer.last_outputs;
}

inline std::vector<std::vector<float>>
forward_network(std::vector<NetworkLayer> &network,
                const std::vector<std::vector<float>> &X) {
  std::vector<std::vector<float>> current_features = X;

  for (size_t l = 0; l < network.size(); l++) {
    bool is_last_layer = (l == network.size() - 1);
    current_features =
        forward_layer(network[l], current_features, !is_last_layer);

    if (is_last_layer) {
      size_t m = current_features.size();
      for (size_t i = 0; i < m; i++) {
        float max_z = network[l].last_zs[i][0];
        for (float z_val : network[l].last_zs[i]) {
          if (z_val > max_z) {
            max_z = z_val;
          }
        }

        float sum_exp = 0;
        for (size_t n = 0; n < network[l].neurons.size(); n++) {
          network[l].last_outputs[i][n] = exp(network[l].last_zs[i][n] - max_z);
          sum_exp += network[l].last_outputs[i][n];
        }

        for (size_t n = 0; n < network[l].neurons.size(); n++) {
          network[l].last_outputs[i][n] /= sum_exp;
        }
      }
      current_features = network[l].last_outputs;
    }
  }
  return current_features;
}

inline void backward_network(std::vector<NetworkLayer> &network,
                             const std::vector<std::vector<float>> &y_true) {
  size_t num_layers = network.size();
  size_t m = y_true.size();

  std::vector<std::vector<float>> current_deltas(m);

  NetworkLayer &output_layer = network[num_layers - 1];
  size_t num_output_neurons = output_layer.neurons.size();

  for (size_t i = 0; i < m; i++) {
    current_deltas[i].resize(num_output_neurons);
    for (size_t n = 0; n < num_output_neurons; n++) {
      float y_pred = output_layer.last_outputs[i][n];
      current_deltas[i][n] = (1.0 / m) * (y_pred - y_true[i][n]);
    }
  }

  for (int l = static_cast<int>(num_layers) - 1; l >= 0; l--) {
    NetworkLayer &layer = network[l];
    size_t num_neurons = layer.neurons.size();
    size_t num_inputs = layer.neurons[0].w.size();

    for (size_t n = 0; n < num_neurons; n++) {
      layer.neurons[n].dw.assign(num_inputs, 0);
      layer.neurons[n].db = 0;
    }

    for (size_t i = 0; i < m; i++) {
      for (size_t n = 0; n < num_neurons; n++) {
        float delta = current_deltas[i][n];

        layer.neurons[n].db += delta;

        for (size_t j = 0; j < num_inputs; j++) {
          float input_val = layer.last_inputs[i][j];
          layer.neurons[n].dw[j] += delta * input_val;
        }
      }
    }

    if (l == 0) {
      break;
    }

    std::vector<std::vector<float>> next_deltas(
        m, std::vector<float>(num_inputs, 0));
    NetworkLayer &prev_layer = network[l - 1];

    for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < num_inputs; j++) {
        float error_sum = 0;
        for (size_t n = 0; n < num_neurons; n++) {
          error_sum += current_deltas[i][n] * layer.neurons[n].w[j];
        }
        float prev_z = prev_layer.last_zs[i][j];
        next_deltas[i][j] = error_sum * relu_derivative(prev_z);
      }
    }

    current_deltas = next_deltas;
  }
}

inline void update_network_weights(std::vector<NetworkLayer> &network,
                                   float alpha,
                                   float l2_regularisation_strength) {
  for (auto &layer : network) {
    for (auto &neuron : layer.neurons) {
      for (size_t j = 0; j < neuron.w.size(); j++) {
        neuron.w[j] -=
            alpha * (neuron.dw[j] + (l2_regularisation_strength * neuron.w[j]));
      }
      neuron.b -= alpha * neuron.db;
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
  float total_ce = 0;

  for (size_t i = 0; i < m; i++) {
    for (size_t n = 0; n < num_outputs; n++) {
      float pred = std::max(y_pred[i][n], 1e-15f);
      total_ce -= y_true[i][n] * log(pred);
    }
  }
  float average_ce = total_ce / m;

  float weigths_squared_sum = 0;
  for (const auto &layer : network) {
    for (const auto &neuron : layer.neurons) {
      for (float w_val : neuron.w) {
        weigths_squared_sum += w_val * w_val;
      }
    }
  }
  float l2_penalty = 0.5 * l2_regularisation_strength * weigths_squared_sum;

  return average_ce + l2_penalty;
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
    size_t num_layer_neurons = is_last_layer ? num_outputs : num_neurons;
    size_t num_layer_inputs = (l == 0) ? num_features : num_neurons;

    std::normal_distribution<float> dist(0.0f,
                                         std::sqrt(2.0f / num_layer_inputs));

    network[l].neurons.resize(num_layer_neurons);

    for (size_t i = 0; i < num_layer_neurons; i++) {
      network[l].neurons[i].w.resize(num_layer_inputs);

      for (size_t j = 0; j < num_layer_inputs; j++) {
        network[l].neurons[i].w[j] = dist(gen);
      }
      network[l].neurons[i].b = 0.0f;
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
