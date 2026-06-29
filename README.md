# whitebox-ml

**A self-contained, header-only C++ library of classical machine learning algorithms, implemented from scratch with zero external dependencies.**

Every algorithm is written entirely in plain C++ using only the C++ Standard Library. There is no dependency on BLAS, LAPACK, Eigen, Armadillo, or any other numerical library. No build system is required — each header can be dropped into any C++ project and compiled directly. Every line of code is visible, auditable, and designed to be understood.

---

## Table of Contents

- [Philosophy](#philosophy)
- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Algorithms](#algorithms)
  - [Simple Linear Regression](#1-simple-linear-regression-gradient-descent)
  - [Closed-Form Linear Regression](#2-closed-form-linear-regression-normal-equation)
  - [Logistic Regression (Binary Classification)](#3-logistic-regression-binary-classification)
  - [L2-Regularised Linear Regression (Ridge Regression)](#4-l2-regularised-linear-regression-ridge-regression)
  - [L2-Regularised Logistic Regression](#5-l2-regularised-logistic-regression)
  - [L2-Regularised Softmax Regression (Multiclass)](#6-l2-regularised-softmax-regression-multiclass-classification)
  - [Support Vector Machine (Binary, Linear)](#7-support-vector-machine-binary-linear)
  - [Multi-Layer Neural Network (Regression)](#8-multi-layer-neural-network-regression)
  - [Multi-Layer Neural Network (Softmax Classification)](#9-multi-layer-neural-network-softmax-classification)
- [Preprocessing Utilities](#preprocessing-utilities)
  - [Standardisation (Z-Score)](#standardisation-z-score)
  - [Normalisation (Min-Max)](#normalisation-min-max)
  - [One-Hot Encoding](#one-hot-encoding)
- [Compilation Guide](#compilation-guide)
- [Code Map](#code-map)
- [Edge Cases & Robustness](#edge-cases--robustness)
- [Numerical Stability](#numerical-stability)
- [Performance Notes](#performance-notes)
- [Project History](#project-history)
- [Roadmap](#roadmap)
- [License](#license)

---

## Philosophy

whitebox-ml is built on a single premise: **machine learning algorithms should be readable**. Many ML libraries are black boxes — they abstract away the math behind layers of optimised BLAS calls and template metaprogramming. While that makes them fast, it makes them terrible for learning.

This library inverts that trade-off. Every implementation is:

- **Header-only**: `#include` what you need, nothing more. No linking, no shared objects, no build system.
- **Self-contained**: Each `.hpp` file stands alone (or depends only on other `.hpp` files in the same project).
- **Educationally transparent**: The code mirrors the mathematical notation directly. You can see the gradient descent loop, the softmax numerator, the hinge loss margin check — in plain sight.
- **Zero-dependency**: Only `<vector>`, `<cmath>`, `<algorithm>`, `<random>`, `<numeric>`, `<iostream>`, `<cstdio>`, `<cstddef>`, `<stdexcept>`, `<utility>`, `<iterator>`, and `<fstream>` are used. Nothing from outside the C++ Standard Library.
- **C++17 compatible**: Uses structured bindings, `auto` return types, and modern C++ idioms.

> **Why not Python?** Python is the lingua franca of ML, but C++ offers a unique perspective: you manage memory explicitly, you see every loop, there is no "magic" from numpy broadcasting or autograd. This makes C++ the best language for *understanding* what the computer actually does when it trains a model.

---

## Features

| Feature | Status |
|---|---|
| Simple linear regression (single feature, gradient descent) | ✅ Complete |
| Multi-feature linear regression (normal equation, closed-form) | ✅ Complete |
| Binary logistic regression (gradient descent) | ✅ Complete |
| L2-regularised linear regression (Ridge regression) | ✅ Complete |
| L2-regularised logistic regression | ✅ Complete |
| L2-regularised softmax regression (multiclass, C classes) | ✅ Complete |
| Linear support vector machine (hinge loss, L2 penalty) | ✅ Complete |
| Multi-layer neural network (regression, ReLU, backpropagation) | ✅ Complete |
| Multi-layer neural network (classification, Softmax, ReLU, backpropagation) | ✅ Complete |
| Z-score standardisation (single value, vector, matrix) | ✅ Complete |
| Min-max normalisation | ✅ Complete |
| One-hot encoding (character data) | ✅ Complete |
| Decision trees (ID3 formulation) | 🔜 Planned |
| Confusion matrix / accuracy metrics | 🔜 Planned |

---

## Architecture

### Header-Only Design

Every algorithm is implemented entirely in `.hpp` files. Functions are declared `inline` to prevent multiple-definition linker errors when the header is included from multiple translation units. This means:

```
#include "logistic-regression.hpp"  // That's it. No .cpp to compile.
```

Each `.cpp` file in the repository is **not** part of the library — it is a self-contained demo/test program that `#include`s the relevant header and exercises the API.

### Data Representation

All data is represented with `std::vector<float>` or `std::vector<std::vector<float>>`:

- **Single feature vector**: `std::vector<float>` — one element per training example.
- **Multi-feature matrix**: `std::vector<std::vector<float>>` — outer vector is examples, inner vector is features (row-major).
- **Labels**: `std::vector<float>` — one element per training example.
- **Multiclass labels**: `float` values in `{0, 1, ..., C-1}` cast from integer class indices.

There is no `Matrix` or `Dataset` wrapper class. Data is passed as plain vectors. This avoids abstraction overhead and keeps the mathematical operations visible.

### Weight Storage

Weights are stored as flat `std::vector<float>`:

- **Single-feature models**: `w` is a single `float`, `b` is a single `float`.
- **Multi-feature models**: `w` is a vector of length `d` (number of features).
- **Softmax models**: `w` is a vector of length `C * d` (flattened weight matrix). The weight for class `c` and feature `j` is at index `c * d + j`.
- **Neural network**: Weights are distributed across `NetworkNeuron` structs within `NetworkLayer` structs.

### Return Convention

Functions that compute updated parameters return them as a `std::vector<float>` (or a struct). The idiomatic calling pattern is:

```cpp
auto [w, b] = train(X, y, w_init, b_init, alpha, lambda, epochs);
```

### Absence of an Umbrella Header

There is no `whitebox-ml.hpp` that includes everything. You should include only the headers you need. This keeps compilation fast and makes dependencies explicit in your source files.

---

## Quick Start

### Prerequisites

- A C++17-compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- No additional libraries or package managers

### Compile and Run a Demo

```bash
# Clone the repository
git clone https://github.com/rxdx6/whitebox-ml.git
cd whitebox-ml

# Compile any demo file directly
g++ -std=c++17 -O2 -o logistic-demo logistic-regression.cpp
./logistic-demo
```

### Use in Your Own Project

```cpp
// my_program.cpp
#include "path/to/whitebox-ml/logistic-regression.hpp"
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> hours_studied = {1, 2, 3, 4, 5, 6};
    std::vector<float> pass_fail = {0, 0, 0, 1, 1, 1};

    auto [w, b] = train(hours_studied, pass_fail, 0.0f, 0.0f, 0.1f, 10000);

    float prob = predict(4.5f, w, b);
    printf("Probability of passing (4.5 hours): %.4f\n", prob);
    return 0;
}
```

```bash
g++ -std=c++17 -O2 -I/path/to/whitebox-ml -o my_program my_program.cpp
./my_program
```

---

## Algorithms

---

### 1. Simple Linear Regression (Gradient Descent)

**File**: `linear-regression.hpp` | **Demo**: `linear-regression.cpp`

#### Mathematical Formulation

Given a single input feature `X` and target `y`, the model is:

```
ŷ = w·x + b
```

**Mean Squared Error (MSE) loss**:

```
J(w, b) = (1/m) * Σᵢ (yᵢ - ŷᵢ)²
```

where `m` is the number of training examples.

**Partial derivatives**:

```
∂J/∂w = (-2/m) * Σᵢ xᵢ * (yᵢ - (w·xᵢ + b))
∂J/∂b = (-2/m) * Σᵢ (yᵢ - (w·xᵢ + b))
```

**Gradient descent update** (repeat for `num_epochs`):

```
w := w - α · ∂J/∂w
b := b - α · ∂J/∂b
```

#### API Reference

```cpp
// Perform one gradient descent step. Returns {updated_w, updated_b}.
std::vector<float> update_w_and_b(
    const std::vector<float>& X,   // Input features (m examples)
    const std::vector<float>& y,   // Target values (m examples)
    float w,                       // Current weight
    float b,                       // Current bias
    float alpha                    // Learning rate
);

// Compute mean squared error over all examples.
float avg_loss(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w,
    float b
);

// Train the model. Returns {trained_w, trained_b}.
std::vector<float> train(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w,              // Initial weight (typically 0.0f)
    float b,              // Initial bias (typically 0.0f)
    float alpha,          // Learning rate
    int num_epochs        // Number of gradient descent iterations
);

// Predict a single value.
float predict(
    float x,   // Input
    float w,   // Trained weight
    float b    // Trained bias
);
```

#### Full Worked Example

```cpp
#include "linear-regression.hpp"
#include <cstdio>

int main() {
    // Toy dataset: 4 points roughly following y = 2x + 1
    std::vector<float> X = {37.8, 39.3, 45.9, 41.3};
    std::vector<float> y = {22.1, 10.4, 9.3, 18.5};

    // Train: start at w=0, b=0, lr=0.0001, 15000 iterations
    auto [w, b_ret] = /* unpack */ train(X, y, 0.0f, 0.0f, 0.0001f, 15000);

    // w_and_b is a vector<float> of size 2 — we can also index
    // float w = w_and_b[0];
    // float b = w_and_b[1];

    float pred = predict(23.0f, w, b_ret);
    printf("Prediction for x=23: y=%.4f\n", pred);
    return 0;
}
```

The `train` function returns weights as a `std::vector<float>{w, b}`. The caller must unpack it.

#### Implementation Details

- The gradient accumulates over the entire training set in each epoch (batch gradient descent — not stochastic or mini-batch).
- The `alpha` value must be small (e.g., `0.0001`) because the features (in the demo) are on the order of 10–50 and the unnormalised MSE gradient is large. Feature scaling (see [Standardisation](#standardisation-z-score)) is strongly recommended for faster convergence.
- No convergence criterion is used — the loop always runs for `num_epochs` iterations.
- The `avg_loss` function is available for manual diagnostic printing, but `train()` does not print by default.

---

### 2. Closed-Form Linear Regression (Normal Equation)

**File**: `closed-form-linear-regression.hpp` | **Demo**: `closed-form-linear-regression.cpp`

#### Mathematical Formulation

For multi-feature linear regression, the normal equation solves for the optimal weights directly without iteration:

```
w* = (Xᵀ · X)⁻¹ · Xᵀ · y
```

where `X` is the design matrix (rows = examples, columns = features + bias column of 1s), and `y` is the target vector.

This is the **exact** minimiser of the MSE loss for linear models — no learning rate, no epochs. However, computing `(Xᵀ · X)⁻¹` is O(n³) in the number of features (due to matrix inversion via Gaussian elimination), making it impractical for high-dimensional data.

#### Implementation Approach

Rather than explicitly computing the matrix inverse, the implementation constructs the augmented matrix `[Xᵀ·X | Xᵀ·y]` and solves the linear system using **Gaussian elimination with partial pivoting**:

1. Construct `X_design` by appending a column of 1s to the input matrix (for the bias term).
2. Compute `Xᵀ · X` (a `d×d` matrix) and `Xᵀ · y` (a `d×1` vector).
3. Build the augmented matrix `[Xᵀ·X | Xᵀ·y]`.
4. For each column `k`, find the row with the largest absolute pivot value (partial pivoting).
5. Swap that row to position `k`.
6. Divide row `k` by the pivot value.
7. Eliminate column `k` from all other rows (Gauss–Jordan style).
8. Extract the solution from the last column of the augmented matrix.

#### API Reference

```cpp
// Matrix transpose: returns A^T (cols x rows).
std::vector<std::vector<float>> matrix_transpose(
    const std::vector<std::vector<float>>& A
);

// Matrix multiplication: returns A * B.
// Throws std::invalid_argument if dimensions don't match.
std::vector<std::vector<float>> matrix_multiplication(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B
);

// Solve linear system A * w = B using Gaussian elimination with partial pivoting.
// A is n×n, B is n×1. Returns the solution vector w of length n.
// Throws std::runtime_error if A is singular or near-singular.
std::vector<float> solve_linear_system(
    const std::vector<std::vector<float>>& A,
    const std::vector<std::vector<float>>& B
);

// Train a linear regression model using the normal equation.
// Returns the weight vector (last element is the bias/intercept).
// Throws std::invalid_argument if X and y have different row counts.
std::vector<float> linear_regressor(
    std::vector<std::vector<float>> X,
    const std::vector<float>& y
);
```

#### Full Worked Example

```cpp
#include "closed-form-linear-regression.hpp"
#include <iostream>

int main() {
    // Data: y = 2x + 1  (approximately)
    std::vector<std::vector<float>> X = {{1.0f}, {2.0f}, {3.0f}, {4.0f}};
    std::vector<float> y = {3.0f, 5.0f, 7.0f, 9.0f};

    std::vector<float> weights = linear_regressor(X, y);

    // weights[0] = slope (≈ 2.0)
    // weights[1] = intercept (≈ 1.0)
    std::cout << "Slope: " << weights[0] << "\n";
    std::cout << "Intercept: " << weights[1] << "\n";

    // Predict: y = weights[0] * 5 + weights[1]  (expects 11.0)
    float pred = weights[0] * 5.0f + weights[1];
    std::cout << "Prediction at x=5: " << pred << "\n";
}
```

#### Key Implementation Details

- The bias term is handled by appending `1.0f` to every row of `X` inside `linear_regressor`. This is the standard design-matrix approach.
- `solve_linear_system` implements the Gauss–Jordan elimination variant (full elimination, not just forward elimination + back-substitution). The pivot is tested against `1e-7f` to detect singular matrices.
- Partial pivoting (selecting the row with the largest absolute value in the current column) improves numerical stability but does not guarantee stability for severely ill-conditioned matrices.
- The matrix functions are standalone utilities available for any purpose, not just regression.
- Empty matrix inputs throw `std::invalid_argument`. Singular matrices throw `std::runtime_error`.

#### Computational Complexity

| Step | Complexity |
|---|---|
| `Xᵀ·X` | O(m·d²) |
| `Xᵀ·y` | O(m·d) |
| Gaussian elimination | O(d³) |
| **Total** | **O(m·d² + d³)** |

Advantage over gradient descent: no learning rate tuning, no iteration count, exact solution.
Disadvantage: O(d³) inversion cost; numerical issues with large d.

---

### 3. Logistic Regression (Binary Classification)

**File**: `logistic-regression.hpp` | **Demo**: `logistic-regression.cpp`

#### Mathematical Formulation

For binary classification, logistic regression models the probability that an example belongs to the positive class:

```
P(y=1|x) = σ(w·x + b)    where σ(z) = 1 / (1 + e⁻ᶻ)
```

The decision boundary is at `P = 0.5`, equivalently `w·x + b = 0`.

**Binary cross-entropy loss** (also called log loss):

```
J(w, b) = -(1/m) * Σᵢ [yᵢ · log(ŷᵢ) + (1 - yᵢ) · log(1 - ŷᵢ)]
```

where `ŷᵢ = σ(w·xᵢ + b)`.

**Gradients**:

```
∂J/∂w = (1/m) * Σᵢ (ŷᵢ - yᵢ) · xᵢ
∂J/∂b = (1/m) * Σᵢ (ŷᵢ - yᵢ)
```

Note the subtle difference from linear regression: the gradient for logistic regression does not have the factor of 2 that appears in MSE, and the prediction is passed through the sigmoid.

#### API Reference

```cpp
// Sigmoid activation function. σ(z) = 1 / (1 + exp(-z)).
float sigmoid_function(float z);

// Predict probability for a single example. Returns σ(w*x + b).
float predict(float x, float w, float b);

// One gradient descent step. Returns {updated_w, updated_b}.
std::vector<float> update_w_and_b(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b, float alpha
);

// Binary cross-entropy loss with epsilon clipping for numerical stability.
float avg_loss(
    const std::vector<float>& X,
    const std::vector<float>& y, float w, float b
);

// Train the model. Prints loss every 1000 epochs. Returns {w, b}.
std::vector<float> train(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b, float alpha, int num_epochs
);
```

#### Full Worked Example

```cpp
#include "logistic-regression.hpp"
#include <cstdio>

int main() {
    // Hours studied -> pass(1) or fail(0)
    std::vector<float> hours = {1, 2, 3, 4, 5, 6};
    std::vector<float> labels = {0, 0, 0, 1, 1, 1};

    auto [w, b] = train(hours, labels, 0.0f, 0.0f, 0.1f, 10000);

    float prob = predict(5.5f, w, b);
    printf("P(pass | 5.5 hrs) = %.4f\n", prob);     // > 0.5
    printf("Class: %c\n", (prob >= 0.5f) ? 'P' : 'F');
}
```

Expected output:

```
Epoch 0 - Current Log Loss: 0.693147
Epoch 1000 - Current Log Loss: 0.465382
...
Epoch 10000 - Current Log Loss: 0.210750
Training complete. Final Log Loss: 0.210750
Prediction for x=5.5: Probability of passing = 0.869245
```

#### Implementation Details

- **Numerical clipping**: In `avg_loss`, the predicted probability is clamped to `[ε, 1-ε]` where ε = 1e⁻⁷. This prevents taking `log(0)`, which would produce `-inf` or `NaN`.
- **Training output**: The `train` function calls `printf` every 1000 epochs to show the current log loss, and a final message when complete. This uses `<cstdio>` without explicitly including it in the header (a minor oversight — relies on `<cmath>` or the including file to transitively include it).
- The model supports only **single-feature** binary classification. For multi-feature logistic regression with L2 regularisation, see [L2-Regularised Logistic Regression](#5-l2-regularised-logistic-regression).

---

### 4. L2-Regularised Linear Regression (Ridge Regression)

**File**: `l2-regularised-linear-regression.hpp` | **Demo**: `l2-regularised-linear-regression.cpp`

#### Mathematical Formulation

Ridge regression adds an L2 penalty to the MSE loss to shrink weights and prevent overfitting:

```
J(w, b) = (1/m) * Σᵢ (yᵢ - ŷᵢ)² + (λ/2) * Σⱼ wⱼ²
```

where λ is the regularisation strength (`l2_regularisation_strength`).

**Gradients** (note the L2 term is NOT applied to the bias `b`):

```
∂J/∂wⱼ = (-2/m) * Σᵢ xᵢⱼ · (yᵢ - ŷᵢ) + λ · wⱼ
∂J/∂b = (-2/m) * Σᵢ (yᵢ - ŷᵢ)
```

**Update rule**:

```
wⱼ := wⱼ - α · [(1/m) · dwⱼ + λ · wⱼ]
b := b - α · (1/m) · db
```

#### API Reference

```cpp
struct LinearRegressionResult {
    std::vector<float> w;  // Weight vector (one per feature)
    float b;               // Bias
};

LinearRegressionResult update_w_and_b(
    const std::vector<std::vector<float>>& X,  // Feature matrix (m x d)
    const std::vector<float>& y,               // Targets (m)
    const std::vector<float>& w,               // Current weights (d)
    float b,                                   // Current bias
    float alpha,                               // Learning rate
    float l2_regularisation_strength           // λ
);

float avg_loss(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& y,
    const std::vector<float>& w,
    float b,
    float l2_regularisation_strength
);

LinearRegressionResult train(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& y,
    const std::vector<float>& w,     // Initial weights (d)
    float b,                         // Initial bias
    float alpha,
    float l2_regularisation_strength,
    int num_epochs
);

std::vector<float> predict(
    const std::vector<std::vector<float>>& X,  // (m x d)
    const std::vector<float>& w,               // (d)
    float b
);
```

#### Full Worked Example

```cpp
#include "l2-regularised-linear-regression.hpp"
#include "standardisation.hpp"
#include <cstdio>

int main() {
    // 4 examples, 3 features
    std::vector<std::vector<float>> X = {
        {230.1, 37.8, 69.2},
        {44.5, 39.3, 45.1},
        {17.2, 45.9, 69.3},
        {151.5, 41.3, 58.5}
    };
    std::vector<float> y = {22.1, 10.4, 9.3, 18.5};

    // Standardise features and target (critical for convergence)
    auto [X_scaled, x_means, x_stds] = standardise_matrix(X);
    auto [y_scaled, y_mean, y_std] = standardise(y);

    // Initialise weights to zero
    std::vector<float> w_init(X[0].size(), 0.0f);

    // Train with L2 regularisation
    auto [w, b] = train(X_scaled, y_scaled, w_init, 0.0f,
                        0.0001f, 0.01f, 15000);

    // Predict on new data (must standardise with training statistics)
    std::vector<std::vector<float>> X_new = {{180.8, 10.8, 58.4}};
    auto X_new_scaled = standardise_vector(X_new[0], x_means, x_stds);
    std::vector<std::vector<float>> X_new_batch = {X_new_scaled};
    std::vector<float> y_pred_scaled = predict(X_new_batch, w, b);

    // Unstandardise prediction
    float y_pred = unstandardise(y_pred_scaled[0], y_mean, y_std);
    printf("Prediction: %.4f\n", y_pred);
}
```

#### Key Implementation Details

- **Multi-feature support**: Unlike the simple linear regression, this version handles `d` features via nested loops. The inner loop accumulates `ŷ = b + Σ wⱼ·X[i][j]`.
- **No intercept regularisation**: The bias `b` is NOT included in the L2 penalty, following standard practice. You can verify this in `avg_loss` — only `w` contributes to `weights_squared_sum`.
- **Return type**: Uses a named `struct LinearRegressionResult` instead of a bare `std::vector<float>`, making the code self-documenting. This is the first header to introduce a struct for this purpose.
- **Standardisation required**: Features and target should be standardised before training. The demo shows exactly how to do this, and then unstandardises the predictions for interpretation.

---

### 5. L2-Regularised Logistic Regression

**File**: `l2-regularised-logistic-regression.hpp` | **Demo**: `l2-regularised-logistic-regression.cpp`

#### Mathematical Formulation

Combines binary logistic regression with L2 regularisation:

```
J(w, b) = -(1/m) * Σᵢ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)] + (λ/2) · w²
```

**Gradients**:

```
∂J/∂w = (1/m) * Σᵢ (ŷᵢ - yᵢ) · xᵢ + λ · w
∂J/∂b = (1/m) * Σᵢ (ŷᵢ - yᵢ)
```

**Update rule**:

```
w := w - α · [(1/m) · dw + λ · w]
b := b - α · (1/m) · db
```

#### API Reference

```cpp
float sigmoid_function(float z);
float predict(float x, float w, float b);

std::vector<float> update_w_and_b(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b, float alpha,
    float l2_regularisation_strength
);

float avg_loss(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b,
    float l2_regularisation_strength
);

std::vector<float> train(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b, float alpha,
    float l2_regularisation_strength,
    int num_epochs
);
```

#### Full Worked Example

```cpp
#include "l2-regularised-logistic-regression.hpp"
#include "standardisation.hpp"

int main() {
    std::vector<float> hours = {1, 2, 3, 4, 5, 6};
    std::vector<float> labels = {0, 0, 0, 1, 1, 1};

    // Standardise the feature
    auto [hours_scaled, mean, stddev] = standardise(hours);

    auto [w, b] = train(hours_scaled, labels, 0.0f, 0.0f,
                        0.1f, 0.01f, 10000);

    // Predict for 5.5 hours (must standardise with training stats)
    float prob = predict(
        standardise(5.5f, mean, stddev), w, b);
    printf("P(pass | 5.5 hrs) = %.4f\n", prob);
}
```

#### Implementation Details

- Same structure as unregularised logistic regression, but the weight update includes the `l2_regularisation_strength * w` term.
- Regularisation is applied only to `w`, not `b`.
- The loss function includes the `(λ/2) · w²` penalty, printed in the training output.
- The clipping trick (`std::max(epsilon, std::min(1-epsilon, y_pred))`) is used in `avg_loss` to prevent `log(0)`.

---

### 6. L2-Regularised Softmax Regression (Multiclass Classification)

**File**: `l2-regularised-softmax-regression.hpp` | **Demo**: `l2-regularised-softmax-regression.cpp`

#### Mathematical Formulation

Softmax regression (multinomial logistic regression) generalises binary logistic regression to `C` classes.

For an input `x` (length `d`) and weights `W` (a `C × d` matrix) and biases `b` (length `C`), the probability of class `c` is:

```
P(y=c|x) = exp(z_c) / Σₖ exp(zₖ)
```

where `z_c = W_c · x + b_c` and `z_k = W_k · x + b_k` for `k = 0, ..., C-1`.

The loss for a single example with true class `y` is the negative log-likelihood:

```
L = -log(P(y| x, W, b))
```

The full cross-entropy loss with L2 regularisation:

```
J(W, b) = -(1/m) * Σᵢ log(ŷᵢ[true_class]) + (λ/2) * Σⱼ Σₖ Wₖⱼ²
```

**Gradient** for class `c`, feature `j`:

```
∂J/∂W_cⱼ = (1/m) * Σᵢ (ŷᵢ[c] - [yᵢ == c]) · xᵢⱼ + λ · W_cⱼ
∂J/∂b_c = (1/m) * Σᵢ (ŷᵢ[c] - [yᵢ == c])
```

#### Weight Storage Convention

Weights are stored in a flat `std::vector<float>` of length `C * d`. The weight for class `c` and feature `j` is at index:

```
index = c * d + j
```

This is a 1D flattening of the `C × d` weight matrix.

#### API Reference

```cpp
struct SoftmaxRegressionResult {
    std::vector<float> w;  // Flattened weight matrix (c * d elements)
    std::vector<float> b;  // Bias vector (c elements)
};

// Predict class probabilities for a single example.
std::vector<float> predict(
    const std::vector<float>& x,  // Input features (d)
    const std::vector<float>& w,  // Flattened weights (c*d)
    const std::vector<float>& b   // Biases (c)
);

SoftmaxRegressionResult update_w_and_b(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& y,           // Integer class labels {0..C-1}
    const std::vector<float>& w,
    const std::vector<float>& b,
    float alpha,
    float l2_regularisation_strength
);

float avg_loss(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& y,
    const std::vector<float>& w,
    const std::vector<float>& b,
    float l2_regularisation_strength
);

SoftmaxRegressionResult train(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& y,
    const std::vector<float>& initial_w,
    const std::vector<float>& initial_b,
    float alpha,
    float l2_regularisation_strength,
    int num_epochs
);
```

#### Full Worked Example

```cpp
#include "l2-regularised-softmax-regression.hpp"
#include "standardisation.hpp"
#include <algorithm>

int main() {
    // 6 examples, 2 features, 3 classes
    std::vector<std::vector<float>> X = {
        {10, 5}, {20, 6}, {30, 7},
        {40, 8}, {50, 9}, {60, 10}
    };
    std::vector<float> y = {0, 0, 1, 1, 2, 2};

    // Standardise features
    auto [X_scaled, means, stds] = standardise_matrix(X);

    // Initialise: 3 classes * 2 features = 6 weights, 3 biases
    std::vector<float> w_init(6, 0.0f);
    std::vector<float> b_init(3, 0.0f);

    auto [w, b] = train(X_scaled, y, w_init, b_init,
                        0.1f, 0.01f, 10000);

    // Predict for a new point
    std::vector<float> x_new = {55, 9.5f};
    std::vector<float> x_new_scaled = standardise_vector(x_new, means, stds);
    std::vector<float> probs = predict(x_new_scaled, w, b);

    for (size_t i = 0; i < probs.size(); i++)
        printf("P(class %zu) = %.4f\n", i, probs[i]);

    int decision = std::distance(probs.begin(),
        std::max_element(probs.begin(), probs.end()));
    printf("Predicted class: %d\n", decision);
}
```

#### Numerical Stability: The Logit-Max Trick

Softmax is numerically unstable when computed naively because `exp(z)` can overflow for large `z` (e.g., `exp(1000)` exceeds the range of `float`). The implementation uses the standard stabilisation trick:

```
z_c' = z_c - max(z₀, ..., z_C₋₁)
softmax(z)_c = exp(z_c') / Σₖ exp(z_k')
```

Since all values are shifted by the same constant, the softmax output is mathematically identical, but all exponentials are computed with non-positive arguments, preventing overflow.

#### Implementation Details

- Labels are passed as `std::vector<float>` but interpreted as integer class indices via `static_cast<int>`.
- The `is_target_class` variable creates a one-hot indicator on the fly: `1.0f` if the class index matches the true label, `0.0f` otherwise.
- Weights are regularised with L2, biases are not.
- The loss function uses only the predicted probability of the true class (negative log-likelihood), unlike binary cross-entropy which sums over both classes.

---

### 7. Support Vector Machine (Binary, Linear)

**File**: `support-vector-machine.hpp` | **Demo**: `support-vector-machine.cpp`

#### Mathematical Formulation

A linear SVM finds the hyperplane `w·x + b = 0` that maximises the margin between two classes. The labels are in `{-1, 1}`.

**Hinge loss** with L2 regularisation:

```
J(w, b) = (λ/2) · w² + (1/m) * Σᵢ max(0, 1 - yᵢ · (w·xᵢ + b))
```

An example contributes to the loss only if it lies inside or on the wrong side of the margin (i.e., `yᵢ · (w·xᵢ + b) < 1`).

**Subgradient** (since hinge loss is not differentiable at the kink):

```
∂J/∂w = λ · w + (1/m) * Σᵢ [ -yᵢ · xᵢ if yᵢ · (w·xᵢ + b) < 1 else 0 ]
∂J/∂b = (1/m) * Σᵢ [ -yᵢ if yᵢ · (w·xᵢ + b) < 1 else 0 ]
```

**Update rule**:

```
w := w - α · [ λ · w + dw/m ]
b := b - α · [ db/m ]
```

#### API Reference

```cpp
std::vector<float> update_w_and_b(
    const std::vector<float>& X,   // Features (m)
    const std::vector<float>& y,   // Labels in {-1, 1} (m)
    float w, float b, float alpha,
    float l2_regularisation_strength
);

float avg_loss(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b,
    float l2_regularisation_strength
);

std::vector<float> train(
    const std::vector<float>& X,
    const std::vector<float>& y,
    float w, float b, float alpha,
    float l2_regularisation_strength,
    int num_epochs
);

// Returns +1 or -1 based on the sign of (w*x + b).
float predict(float x, float w, float b);
```

#### Full Worked Example

```cpp
#include "support-vector-machine.hpp"
#include <cstdio>

int main() {
    // Labels must be -1 and +1 (not 0 and 1)
    std::vector<float> hours = {1, 2, 3, 4, 5, 6};
    std::vector<float> labels = {-1, -1, -1, 1, 1, 1};

    auto [w, b] = train(hours, labels, 0.0f, 0.0f,
                        0.05f, 0.01f, 5000);

    printf("Decision boundary at x = %.4f\n", -b / w);

    float pred = predict(3.5f, w, b);
    printf("Prediction for x=3.5: label = %.0f\n", pred);
    // pred will be -1 (fail) or 1 (pass)
}
```

#### Implementation Details

- **Label convention**: Unlike logistic regression (which uses `{0, 1}`), the SVM uses `{-1, 1}` labels because the margin condition `yᵢ · (w·xᵢ + b) >= 1` requires signed labels.
- **Margin check**: The gradient only accumulates from examples where `yᵢ * ((w * X[i]) + b) < 1` — these are the support vectors-in-training. Examples beyond the margin contribute zero gradient.
- **Subgradient framework**: The implementation uses the subgradient of the hinge loss at the non-differentiable point `margin = 1`. This is standard practice for SVM training with gradient descent.
- **Decision boundary**: The boundary value is `x = -b/w`. This is the point where `w·x + b = 0`.

---

### 8. Multi-Layer Neural Network (Regression)

**File**: `linear_regression_neural_network.hpp` | **Demo**: `linear_regression_neural_network.cpp`

#### Architecture

This is a fully-connected feedforward network supporting:

- An arbitrary number of hidden layers (specified by `num_layers`)
- An arbitrary number of neurons per hidden layer (specified by `num_neurons`, applied uniformly)
- ReLU activation on all hidden layers
- Linear (no activation) on the output layer
- MSE loss with L2 regularisation
- Batch gradient descent with full backpropagation

The network architecture is:

```
Input (d features)
  → [Linear + ReLU] (num_neurons)
    → [Linear + ReLU] (num_neurons)
      → ... (num_layers - 2 more hidden layers)
        → Linear (num_outputs)
```

#### Data Structures

```cpp
struct NetworkNeuron {
    std::vector<float> w;  // Weight vector (one per input to this neuron)
    float b;               // Bias
    std::vector<float> dw; // Accumulated weight gradient
    float db;              // Accumulated bias gradient
};

struct NetworkLayer {
    std::vector<NetworkNeuron> neurons;

    // Cached values for backpropagation:
    std::vector<std::vector<float>> last_inputs;   // Input to this layer (m x input_dim)
    std::vector<std::vector<float>> last_zs;       // Pre-activation values (m x neuron_count)
    std::vector<std::vector<float>> last_outputs;  // Post-activation values (m x neuron_count)
};
```

The network itself is `std::vector<NetworkLayer>`.

#### Forward Pass

For each layer `l` with `n_l` neurons:

1. For each example `i` and neuron `n`:
   ```
   z_{i,n} = b_n + Σⱼ w_{n,j} · a_{i,j}ᵠ⁻¹
   ```
   where `aᵠ⁻¹` is the output of the previous layer (or the raw input for layer 0).

2. Apply activation:
   - Hidden layers: `a_{i,n} = ReLU(z_{i,n}) = max(0, z_{i,n})`
   - Output layer: `a_{i,n} = z_{i,n}` (linear)

3. All `last_inputs`, `last_zs`, and `last_outputs` are cached for the backward pass.

#### Backward Pass (Backpropagation)

**Output layer delta** (MSE gradient):

```
δₙᵒᵘᵗ = (2/m) · (ŷᵢ,ₙ - yᵢ,ₙ)
```

**Hidden layer delta** (backpropagated through ReLU):

```
δⱼʰⁱᵈᵈᵉⁿ = (Σₙ δₙ · w_{n,j}) · ReLU'(zⱼ)
```

where `ReLU'(z) = 1 if z > 0 else 0`.

**Gradient accumulation**:

```
∂J/∂w_{n,j} = Σᵢ δᵢ,ₙ · aᵢ,ⱼᵠ⁻¹
∂J/∂b_n = Σᵢ δᵢ,ₙ
```

#### Weight Initialisation (He Initialisation)

Weights are initialised using He (Kaiming) initialisation, designed for ReLU networks:

```
W ~ N(0, √(2 / n_in))
```

where `n_in` is the number of inputs to the layer. The random number generator uses a fixed seed (`42`) for reproducibility.

#### API Reference

```cpp
// Forward pass through a single layer.
std::vector<std::vector<float>> forward_layer(
    NetworkLayer& layer,
    const std::vector<std::vector<float>>& input_batch,
    bool apply_activation
);

// Forward pass through the entire network.
std::vector<std::vector<float>> forward_network(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& X
);

// Backward pass through the entire network.
void backward_network(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& y_true
);

// Apply gradient descent to all weights and biases.
void update_network_weights(
    std::vector<NetworkLayer>& network,
    float alpha,
    float l2_regularisation_strength
);

// Calculate MSE + L2 loss.
float calculate_network_loss(
    const std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& y_pred,
    const std::vector<std::vector<float>>& y_true,
    float l2_regularisation_strength
);

// Train a pre-existing network.
void train_network(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& y,
    float alpha,
    float l2_regularisation_strength,
    int num_epochs,
    bool print_progress = true
);

// Full pipeline: initialise + train.
std::vector<NetworkLayer> train(
    const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& y,
    int num_layers,
    int num_neurons,
    float alpha,
    float l2_regularisation_strength,
    int num_epochs,
    bool print_progress = true
);

// Predict using a trained network.
std::vector<std::vector<float>> predict(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& X
);
```

#### Full Worked Example (auto-mpg dataset)

```cpp
#include "linear_regression_neural_network.hpp"
#include <fstream>
#include <sstream>
#include <cstdio>

int main() {
    // Load pre-standardised auto-mpg dataset
    std::ifstream file("auto-mpg.csv");
    std::string line;
    std::getline(file, line);  // skip header

    std::vector<std::vector<float>> X, y;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string val;
        std::vector<float> row;
        while (std::getline(ss, val, ','))
            row.push_back(std::stof(val));
        y.push_back({row[0]});              // mpg is first column
        X.push_back({row.begin() + 1, row.end()});  // rest are features
    }

    // Train/test split (50/50)
    size_t split = X.size() / 2;
    std::vector<std::vector<float>> X_train(X.begin(), X.begin() + split);
    std::vector<std::vector<float>> y_train(y.begin(), y.begin() + split);
    std::vector<std::vector<float>> X_test(X.begin() + split, X.end());
    std::vector<std::vector<float>> y_test(y.begin() + split, y.end());

    // 3 hidden layers, 128 neurons each
    auto network = train(X_train, y_train, 3, 128,
                         0.005f, 0.001f, 10000, true);

    auto predictions = predict(network, X_test);
    float loss = calculate_network_loss(network, predictions, y_test, 0.001f);
    printf("Test loss: %.6f\n", loss);
}
```

#### Implementation Details

- **Caching**: Every forward pass stores `last_inputs`, `last_zs`, and `last_outputs` in each layer. This triples memory usage but makes the backward pass straightforward — all intermediate values are available without recomputation.
- **Output layer**: No activation (`apply_activation = false` for the last layer). This is appropriate for regression where the output is unbounded.
- **Delta scaling**: The output delta uses `(2.0 / m)` rather than `(2.0 * (1.0 / m))` — both are equivalent, but the in-line constant avoids an extra division.
- **Printing**: The `train_network` function prints every 100 epochs (plus the first and last epoch) to `std::cout`. This can be suppressed with `print_progress = false`.
- **Fixed random seed**: The `train()` convenience function seeds `std::mt19937` with `42`. Change this for different initialisations.

---

### 9. Multi-Layer Neural Network (Softmax Classification)

**File**: `softmax_regression_neural_network.hpp` | **Demo**: `softmax_regression_neural_network.cpp`

#### Architecture

This is a fully-connected feedforward network supporting:

- An arbitrary number of hidden layers (specified by `num_layers`)
- An arbitrary number of neurons per hidden layer (specified by `num_neurons`, applied uniformly)
- ReLU activation on all hidden layers
- Softmax activation on the output layer (multiclass classification)
- Cross-entropy loss with L2 regularisation
- Batch gradient descent with full backpropagation

The network architecture is:

```
Input (d features)
  → [Linear + ReLU] (num_neurons)
    → [Linear + ReLU] (num_neurons)
      → ... (num_layers - 2 more hidden layers)
        → Softmax (num_outputs)
```

#### Data Structures

```cpp
struct NetworkNeuron {
    std::vector<float> w;  // Weight vector (one per input to this neuron)
    float b = 0;           // Bias
    std::vector<float> dw; // Accumulated weight gradient
    float db = 0;          // Accumulated bias gradient
};

struct NetworkLayer {
    std::vector<NetworkNeuron> neurons;

    // Cached values for backpropagation:
    std::vector<std::vector<float>> last_inputs;   // Input to this layer (m x input_dim)
    std::vector<std::vector<float>> last_zs;       // Pre-activation values (m x neuron_count)
    std::vector<std::vector<float>> last_outputs;  // Post-activation values (m x neuron_count)
};
```

The network itself is represented as `std::vector<NetworkLayer>`.

#### Forward Pass

For each layer `l` with `n_l` neurons:

1. For each example `i` and neuron `n`:
   ```
   z_{i,n} = b_n + Σⱼ w_{n,j} · a_{i,j}ᵠ⁻¹
   ```
   where `aᵠ⁻¹` is the output of the previous layer (or the raw input for layer 0).

2. Apply activation:
   - Hidden layers: `a_{i,n} = ReLU(z_{i,n}) = max(0, z_{i,n})`
   - Output layer: `a_{i,n} = Softmax(z_{i,n}) = exp(z_{i,n} - max_z) / Σₖ exp(z_{i,k} - max_z)` (using logit-max trick to prevent overflow)

3. All `last_inputs`, `last_zs`, and `last_outputs` are cached for the backward pass.

#### Backward Pass (Backpropagation)

**Output layer delta** (Cross-entropy with softmax gradient):

```
δᵢ,ₙᵒᵘᵗ = (1/m) · (ŷᵢ,ₙ - yᵢ,ₙ)
```

where `ŷᵢ,ₙ` is the predicted softmax probability for class `n`, and `yᵢ,ₙ` is the target value (typically one-hot encoded).

**Hidden layer delta** (backpropagated through ReLU):

```
δⱼʰⁱᵈᵈᵉⁿ = (Σₙ δₙ · w_{n,j}) · ReLU'(zⱼ)
```

where `ReLU'(z) = 1 if z > 0 else 0`.

**Gradient accumulation**:

```
∂J/∂w_{n,j} = Σᵢ δᵢ,ₙ · aᵢ,ⱼᵠ⁻¹
∂J/∂b_n = Σᵢ δᵢ,ₙ
```

#### Weight Initialisation (He Initialisation)

Weights are initialised using He (Kaiming) initialisation:

```
W ~ N(0, √(2 / n_in))
```

where `n_in` is the number of inputs to the layer. The random number generator uses a fixed seed (`42`) for reproducibility.

#### API Reference

```cpp
// Forward pass through a single layer.
std::vector<std::vector<float>> forward_layer(
    NetworkLayer& layer,
    const std::vector<std::vector<float>>& input_batch,
    bool apply_activation
);

// Forward pass through the entire network.
std::vector<std::vector<float>> forward_network(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& X
);

// Backward pass through the entire network.
void backward_network(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& y_true
);

// Apply gradient descent to all weights and biases.
void update_network_weights(
    std::vector<NetworkLayer>& network,
    float alpha,
    float l2_regularisation_strength
);

// Calculate cross-entropy + L2 loss.
float calculate_network_loss(
    const std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& y_pred,
    const std::vector<std::vector<float>>& y_true,
    float l2_regularisation_strength
);

// Train a pre-existing network.
void train_network(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& y,
    float alpha,
    float l2_regularisation_strength,
    int num_epochs,
    bool print_progress = true
);

// Full pipeline: initialise + train.
std::vector<NetworkLayer> train(
    const std::vector<std::vector<float>>& X,
    const std::vector<std::vector<float>>& y,
    int num_layers,
    int num_neurons,
    float alpha,
    float l2_regularisation_strength,
    int num_epochs,
    bool print_progress = true
);

// Predict using a trained network.
std::vector<std::vector<float>> predict(
    std::vector<NetworkLayer>& network,
    const std::vector<std::vector<float>>& X
);
```

#### Full Worked Example

```cpp
#include "softmax_regression_neural_network.hpp"
#include "one-hot-encode.hpp"
#include <cstdio>
#include <vector>

int main() {
    // Toy 2D features dataset for a 3-class problem
    std::vector<std::vector<float>> features = {
        {1.5f, 0.2f}, {-0.5f, 2.1f}, {0.1f, -1.2f}, {2.0f, 1.8f}
    };

    std::vector<char> input_data = {'A', 'B', 'C', 'A'};
    std::vector<char> categories = {'A', 'B', 'C'};

    // One-hot encode the categorical targets
    auto encoded_int = encode(input_data, categories);
    std::vector<std::vector<float>> classes;
    for (const auto &row : encoded_int) {
        classes.push_back(std::vector<float>(row.begin(), row.end()));
    }

    // 3 hidden layers, 128 neurons each
    int num_layers = 3;
    int num_neurons = 128;
    float alpha = 0.005f;
    float l2_regularisation_strength = 0.001f;
    int num_epochs = 10000;

    auto network = train(features, classes, num_layers, num_neurons,
                         alpha, l2_regularisation_strength, num_epochs, true);

    auto predictions = predict(network, features);

    printf("\nFinal Output Probabilities:\n");
    for (size_t i = 0; i < predictions.size(); i++) {
        printf("Input %zu -> [ ", i);
        for (float prob : predictions[i]) {
            printf("%f ", prob);
        }
        printf("]\n");
    }
}
```

#### Implementation Details

- **Softmax Output Activation**: The output layer applies the logit-max trick to prevent overflow when computing exponentials for class probabilities.
- **Cross-Entropy Loss**: Evaluates $-\frac{1}{m} \sum y \log(\hat{y})$. Probabilities are clipped to a minimum of `1e-15f` to prevent numerical issues with `log(0)`.
- **Backpropagation Delta**: The output delta calculation matches the softmax cross-entropy gradient: $\frac{1}{m} (\hat{y} - y)$, where $y$ is the target matrix of probabilities (typically one-hot vectors).
- **Caching**: Just like the regression neural network, intermediate inputs, z-values, and outputs are cached in `NetworkLayer` to avoid recomputations during backpropagation.

---

## Preprocessing Utilities

### Standardisation (Z-Score)

**File**: `standardisation.hpp`

Standardisation transforms data to have zero mean and unit variance:

```
z = (x - μ) / σ
```

where `μ` is the mean and `σ` is the standard deviation.

#### API

```cpp
// Standardise or unstandardise a single value.
float standardise(float value, float mean, float standard_deviation);
float unstandardise(float standardised_value, float mean, float standard_deviation);

// Standardise a vector. Returns {scaled_vector, mean, standard_deviation}.
struct StandardiseResult {
    std::vector<float> scaled_vector;
    float mean;
    float standard_deviation;
};
StandardiseResult standardise(const std::vector<float>& x);

// Standardise a matrix (column-wise). Returns {scaled_matrix, means, standard_deviations}.
struct StandardiseMatrixResult {
    std::vector<std::vector<float>> scaled_matrix;
    std::vector<float> means;
    std::vector<float> standard_deviations;
};
StandardiseMatrixResult standardise_matrix(const std::vector<std::vector<float>>& X);

// Apply precomputed means/stds to a new vector.
std::vector<float> standardise_vector(
    const std::vector<float>& x,
    const std::vector<float>& means,
    const std::vector<float>& standard_deviations
);
```

#### Edge Cases

- **Zero standard deviation**: If all values in a vector are identical, `standard_deviation` will be 0. The function returns a vector of all zeros (since `(x - μ) / 0` is undefined).
- **Empty input**: Returns `{{}, 0.0f, 1.0f}` for the vector overload, `{{}, {}, {}}` for the matrix overload.
- **Matrix standardisation**: Processes each column independently, extracting the column into a temporary `std::vector<float>`, standardising it, then writing it back.

### Normalisation (Min-Max)

**File**: `normalisation.hpp`

Normalisation scales data to the range `[0, 1]`:

```
x' = (x - x_min) / (x_max - x_min)
```

#### API

```cpp
std::vector<float> normaliser(std::vector<float> values);
```

#### Edge Cases

- **Zero range** (`x_min == x_max`): All values are set to 0.
- **Empty input**: Returns an empty vector.
- The function modifies the input vector in place (it takes a **copy** since the parameter is passed by value), then returns it.

### One-Hot Encoding

**File**: `one-hot-encode.hpp`

Converts a sequence of categorical characters into a binary matrix representation.

#### API

```cpp
std::vector<std::vector<int>> encode(
    const std::vector<char>& input_data,   // Raw categorical values
    const std::vector<char>& categories    // All possible categories
);
```

#### Behaviour

For each element in `input_data`, the function finds its index in `categories` and produces a vector of length `|categories|` with a 1 at that index and 0 elsewhere. If a value in `input_data` is not found in `categories`, the entire vector remains zeros (the category is silently treated as unknown).

#### Example

```cpp
std::vector<char> data = {'a', 'b', 'a', 'c'};
std::vector<char> cats = {'a', 'b', 'c'};
auto encoded = encode(data, cats);
// Result: {{1,0,0}, {0,1,0}, {1,0,0}, {0,0,1}}
```

---

## Compilation Guide

### Per-File Compilation

Each `.cpp` file is self-contained. To compile any demo:

```bash
# Simple linear regression
g++ -std=c++17 -O2 -o lr linear-regression.cpp && ./lr

# Closed-form (includes matrix operations)
g++ -std=c++17 -O2 -o cflr closed-form-linear-regression.cpp && ./cflr

# Logistic regression
g++ -std=c++17 -O2 -o logit logistic-regression.cpp && ./logit

# L2-regularised linear regression (requires standardisation.hpp)
g++ -std=c++17 -O2 -o ridge l2-regularised-linear-regression.cpp && ./ridge

# L2-regularised logistic regression
g++ -std=c++17 -O2 -o rlogit l2-regularised-logistic-regression.cpp && ./rlogit

# L2-regularised softmax regression
g++ -std=c++17 -O2 -o softmax l2-regularised-softmax-regression.cpp && ./softmax

# Support vector machine
g++ -std=c++17 -O2 -o svm support-vector-machine.cpp && ./svm

# Neural network (requires auto-mpg.csv in working directory)
g++ -std=c++17 -O2 -o nn linear_regression_neural_network.cpp && ./nn

# Classification neural network
g++ -std=c++17 -O2 -o classification_nn softmax_regression_neural_network.cpp && ./classification_nn
```

### Compiler Flags

| Flag | Purpose |
|---|---|
| `-std=c++17` | Enable C++17 features (structured bindings, etc.) |
| `-O2` or `-O3` | Optimisation; the algorithms are loop-heavy and benefit significantly |
| `-march=native` | Enable CPU-specific optimisations (AVX, SSE) on supported compilers |
| `-Wall -Wextra` | Enable warnings for code quality |
| `-Wno-unused-variable` | Some demos have unused variables; suppress if desired |

### Using Precompiled Headers (PCH)

For faster recompilation during development, the repository includes precompiled headers (`.pch` files). These are compiler-specific and may need regeneration:

```bash
g++ -std=c++17 -x c++-header standardisation.hpp
g++ -std=c++17 -x c++-header linear_regression_neural_network.hpp
g++ -std=c++17 -x c++-header softmax_regression_neural_network.hpp
```

### Build Script

There is no Makefile or CMakeLists.txt — the project intentionally avoids build infrastructure. However, a minimal build script for all demos can be written as:

```bash
#!/bin/bash
# build_all.sh
set -e
g++ -std=c++17 -O2 -o build/linear-regression linear-regression.cpp
g++ -std=c++17 -O2 -o build/closed-form-linear-regression closed-form-linear-regression.cpp
g++ -std=c++17 -O2 -o build/logistic-regression logistic-regression.cpp
g++ -std=c++17 -O2 -o build/l2-regularised-linear-regression l2-regularised-linear-regression.cpp
g++ -std=c++17 -O2 -o build/l2-regularised-logistic-regression l2-regularised-logistic-regression.cpp
g++ -std=c++17 -O2 -o build/l2-regularised-softmax-regression l2-regularised-softmax-regression.cpp
g++ -std=c++17 -O2 -o build/support-vector-machine support-vector-machine.cpp
g++ -std=c++17 -O2 -o build/linear-regression-neural-network linear_regression_neural_network.cpp
g++ -std=c++17 -O2 -o build/softmax-regression-neural-network softmax_regression_neural_network.cpp
```

---

## Code Map

| File | Lines | Type | Description |
|---|---|---|---|
| `linear-regression.hpp` | 46 | Header | Simple linear regression via batch gradient descent (single feature) |
| `closed-form-linear-regression.hpp` | 139 | Header | Multi-feature linear regression via normal equation (Gaussian elimination) |
| `logistic-regression.hpp` | 62 | Header | Binary logistic regression with sigmoid and cross-entropy loss |
| `l2-regularised-linear-regression.hpp` | 96 | Header | Multi-feature Ridge regression (MSE + L2) |
| `l2-regularised-logistic-regression.hpp` | 65 | Header | Binary logistic regression with L2 regularisation |
| `l2-regularised-softmax-regression.hpp` | 133 | Header | Multiclass softmax regression (C classes) with L2 regularisation |
| `support-vector-machine.hpp` | 58 | Header | Linear SVM with hinge loss and L2 regularisation |
| `linear_regression_neural_network.hpp` | 237 | Header | Multi-layer feedforward network with ReLU and backpropagation |
| `softmax_regression_neural_network.hpp` | 262 | Header | Multi-layer feedforward network with ReLU, Softmax output, and backpropagation |
| `standardisation.hpp` | 96 | Header | Z-score standardisation (single value, vector, matrix) |
| `normalisation.hpp` | 26 | Header | Min-max normalisation |
| `one-hot-encode.hpp` | 28 | Header | One-hot encoding for character data |
| `linear-regression.cpp` | 12 | Demo | Simple LR on 4 points |
| `closed-form-linear-regression.cpp` | 38 | Demo | Closed-form LR on y=2x+1 toy data |
| `logistic-regression.cpp` | 22 | Demo | Binary classification (hours vs pass/fail) |
| `l2-regularised-linear-regression.cpp` | 33 | Demo | Ridge regression with standardisation |
| `l2-regularised-logistic-regression.cpp` | 26 | Demo | Regularised logistic with standardisation |
| `l2-regularised-softmax-regression.cpp` | 30 | Demo | 3-class softmax on 2D features |
| `support-vector-machine.cpp` | 25 | Demo | Binary SVM classification |
| `linear_regression_neural_network.cpp` | 41 | Demo | 3-layer net on auto-mpg dataset |
| `softmax_regression_neural_network.cpp` | 39 | Demo | 3-layer net classification on toy dataset |
| `decision-trees.cpp` | 3 | Stub | ID3 decision tree (not yet implemented) |
| `auto-mpg.csv` | 393 | Data | Pre-standardised auto MPG dataset (7 features, 1 target) |

---

## Edge Cases & Robustness

### Empty Inputs

| Function | Behaviour |
|---|---|
| `matrix_transpose` | Returns empty matrix if input is empty or has empty rows |
| `matrix_multiplication` | Throws `std::invalid_argument` |
| `linear_regressor` | Throws `std::invalid_argument` |
| `standardise(vector)` | Returns `{{}, 0.0f, 1.0f}` |
| `standardise_matrix(matrix)` | Returns `{{}, {}, {}}` |
| `normaliser` | Returns `{}` |
| `encode` | Returns empty matrix if `input_data` is empty |

### Singular Matrices

`solve_linear_system` checks if the pivot is smaller than `1e-7f` and throws `std::runtime_error` if so. The partial pivoting step (selecting the maximum absolute value in the column) mitigates singular behaviour but does not eliminate it entirely.

### Zero-Variance Data

- `standardise`: If `standard_deviation == 0`, returns a vector of zeros.
- `standardise` (single value): If `standard_deviation == 0`, returns `0.0f`.
- `normaliser`: If `min == max`, fills the vector with zeros.

### Missing Categories (One-Hot Encoding)

If an input value is not found in the provided `categories`, the corresponding one-hot vector is all zeros (a "null" encoding). No error is raised.

### Log of Zero

All loss functions that use `log()` clamp their inputs:
- Logistic/softmax regression: `max(ε, min(1-ε, y_pred))` where ε = 1e⁻⁷
- Softmax Neural Network: `max(y_pred, 1e-15f)`
- SVM: Hinge loss does not use `log()`, so it is unaffected.

---

## Numerical Stability

### Softmax: The Logit-Max Trick

The softmax function (used in both `l2-regularised-softmax-regression.hpp` and `softmax_regression_neural_network.hpp`) is implemented to subtract the maximum logit before computing exponentials. For example, in `l2-regularised-softmax-regression.hpp`:

```cpp
float maximum_logit = -INFINITY;
for (int c = 0; c < num_classes; c++) {
    logits[c] = dot_product + b[c];
    if (logits[c] > maximum_logit) maximum_logit = logits[c];
}
// Subtract max before exp():
hat_y[c] = exp(logits[c] - maximum_logit);
```

In `softmax_regression_neural_network.hpp`, the same principle is applied to the output layers' pre-activation values $z$:

```cpp
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
```

This guarantees all exponents are ≤ 0, preventing `exp(x)` overflow for large logits.

### Floating-Point Precision

All computations use `float` (32-bit IEEE 754). This is sufficient for learning tasks and converges correctly but has limited precision compared to `double`. Key limitations:

- MSE values below `1e-7` may vanish in the presence of larger gradients.
- Weight updates may underflow for very small learning rates (e.g., α < 1e⁻⁷).

If higher precision is needed, replace `float` with `double` throughout the header.

### Random Seed

The neural networks use a fixed seed (`42`) for reproducibility. Every run produces identical initialisation and identical results. To vary initialisation, change the seed or make it configurable.

---

## Performance Notes

### Computational Profile

| Algorithm | Per-Epoch Cost | Convergence |
|---|---|---|
| Simple linear regression | O(m) | ~10⁴ epochs for small data |
| Closed-form (normal equation) | O(m·d² + d³) | Exact (single step) |
| Logistic regression | O(m) | ~10⁴ epochs |
| L2-regularised linear | O(m·d) | ~10⁴ epochs |
| L2-regularised logistic | O(m) | ~10⁴ epochs |
| L2-regularised softmax | O(m·C·d) | ~10⁴ epochs |
| SVM (hinge) | O(m) | ~5×10³ epochs |
| Neural network (Regression) | O(m·L·n²) | ~10⁴ epochs |
| Neural network (Softmax) | O(m·L·n²) | ~10⁴ epochs |

where `m` = examples, `d` = features, `C` = classes, `L` = layers, `n` = neurons/layer.

### Memory Usage

- All data is stored in `std::vector` with no memory pooling.
- The neural networks cache the full forward pass (inputs, z-values, outputs for every example at every layer), which is O(m · L · n). For the regression neural network demo (393 examples, 3 layers, 128 neurons), this is approximately 393 × 3 × 128 × 3 × 4 bytes ≈ 1.8 MB for the caches alone.
- Matrix operations in the closed-form solver allocate temporary matrices (e.g., the augmented matrix of size n × (n+1)).

### Optimisation Suggestions

1. **Pre-standardise** all features and targets. Gradient descent converges orders of magnitude faster on scaled data.
2. **Use `-O3 -march=native`** to enable auto-vectorisation (SSE/AVX) for loop-heavy operations.
3. **Increase batch size** conceptually: all current algorithms use full-batch gradient descent. True "batches" would require a training loop that iterates over subsets.
4. **Reduce epoch count** by monitoring loss convergence and stopping early (not currently implemented).

---

## Project History

The repository's git history reveals the evolution of the project:

1. **`63867c9` — Initial stub**: `feature_vector` and `dataset` data structures.
2. **`713684f` — .gitignore**: Adds ignore rules and removes binaries from version control.
3. **`87a5bcf` — SVM + feature vector enhancements**: First algorithmic implementation (SVM).
4. **`59a08c5` — Rename**: `feature-vector.hpp` becomes `structs.hpp`.
5. **`5d86dd5` — Linear regression + ML stubs**: First linear regression implementation.
6. **`4689dae` — Refactor into headers**: Core algorithms move from `.cpp` to `.hpp`, establishing the header-only pattern.
7. **`7f4551f` — Comment migration**: Documentation moves into headers.
8. **`bed1a68` — standardisation.hpp + PCH**: Adds the preprocessing utility and precompiled headers.
9. **`8d3d0a4` — L2 regularisation**: Ridge regression and standardisation utilities.
10. **`01981d7` — Softmax**: Multiclass classification added.
11. **`3ba4747` — API refactor**: Names become clearer, multi-dim support improved.
12. **`2707ace` — Flatten softmax weights**: Weight matrix flattened to 1D vector.
13. **`f16f374` — Naming simplification**: Headers streamlined.
14. **`1e0f820` — Neural network forward pass**: First NN implementation.
15. **`df87404` — CSV loading + train/test split**: auto-mpg dataset integration.
16. **`03d27e9` — CSV data loading, split, loss**: Refined training pipeline.
17. **`48bb09a` — CSV gitignore**: Dataset ignored from version control.

---

## Roadmap

### High Priority

- [ ] **ID3 Decision Tree** (`decision-trees.cpp`): The stub already exists with the mathematical formulation commented out. The ID3 algorithm selects splits by maximising information gain (or minimising entropy) at each node.
- [ ] **Confusion Matrix / Accuracy / F1**: Metrics for evaluating classifier performance.
- [ ] **k-Fold Cross-Validation**: A utility to split data into folds and iterate.

### Medium Priority

- [ ] **Stochastic / Mini-Batch Gradient Descent**: Replace full-batch with mini-batch for efficiency on larger datasets.
- [ ] **Learning Rate Schedules**: Decay the learning rate over time (step decay, exponential decay).
- [ ] **Additional Activation Functions**: Tanh, sigmoid for hidden layers, leaky ReLU.
- [ ] **Additional Optimisers**: Momentum, RMSprop, Adam (beyond vanilla SGD).

### Lower Priority

- [ ] **L1 Regularisation (Lasso)**: Changes the penalty to `λ · |w|` with subgradient-based updates.
- [ ] **Polynomial Features**: Expand input features to polynomial combinations.
- [ ] **Principal Component Analysis (PCA)**: Dimensionality reduction via eigendecomposition.
- [ ] **k-Means Clustering**: Unsupervised learning.
- [ ] **Confidence Intervals**: Bootstrapping-based uncertainty estimation.
- [ ] **Umbrella Header**: `whitebox-ml.hpp` that includes all algorithm headers.

---

*whitebox-ml: Because the best way to learn ML is to read every line.*
