// Decision boundry is given by wx - b = 0
// where w is a real valued vector,
// where b is a real number,
// where x is our input feature vector,
// where wx = w[1]x[1] + w[2]x[2] + ... + w[D]x[D]
// and D is the number of dimensions of the feature vector x
//
// The predict label y is given by y = sign(wx - b)
// where sign is a mathematical operator that takes any value as input and
// returns +1 if the input is a positive number or -1 if the input is a negative
// number
//
// The goal of SVM is to find w* and b* which are optimal w and b such that
// f(x) = sign(w*x - b*)
//
// wx[i] - b >= 1 if y[i] = +1 and
// wx[i] - b <= 1 if y[i] = -1
//
// We would also prefer that the hyperplane separates positive examples from
// negative ones with the largest margin. The margin is the distance between the
// closest examples of two classes, as defined by the decision boundary. A large
// margin contributes to a better generalization, that is how well the model
// will classify new examples in the future. To achieve that, we need to
// minimize the Euclidean norm of w denoted by ||w|| and given by
// sqrt(
// summation j = 1 to D(
// w[j] * w[j];
// )
// )
//
// SVM: Minimize ||w|| subject to y[i](wx[i] - b) >= 1 for i = 1, ..., N. given
// y[i](w * x[i] - b) >= 1

#include "structs.hpp"
#include <cstddef>

struct svm {
  ml::feature_vector w;
  ml::feature_vector x;
  float b;

  svm(size_t dimensions) : w(dimensions), x(dimensions), b(0) {}

  float wx() {
    float sum = 0;
    for (size_t i = 0; i < w.size(); i++) {
      sum += w[i] * x[i];
    }
    return sum;
  }

  float predict() { return wx() - b; }

  int sign(float input) { return input > 0 ? 1 : -1; }

  int classify() { return sign(predict()); }
};
