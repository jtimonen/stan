#ifndef STAN_SERVICES_UTIL_UNDERSTANDING_STAN
#define STAN_SERVICES_UTIL_UNDERSTANDING_STAN

#include <iostream>
#include <vector>

namespace stan {
namespace services {
namespace util {

/**
 * Print a double vector to std::cout.
 *
 * @param[in] vec the vector to print
 * @return nothing
 */
void print_vector(std::vector<double> vec) {
  std::cout << "[";
  const int L = vec.size();
  for (int i = 0; i < L - 1; ++i) { 
    std::cout << vec[i] << ", ";
  }
  std::cout << vec[L-1] << "]";
}

/**
 * Print a log_prob evaluation to std::cout.
 *
 * @param[in] x the point where log_prob was evaluated
 * @param[in] lp value of log probability
 * @return nothing
 */
void print_log_prob_eval(std::vector<double> x, double lp) {
  std::cout << " * log_prob() eval, x = ";
  print_vector(x);
  std::cout << ", lp = " << lp << "\n";
}

/**
 * Print a log_prob_grad evaluation to std::cout.
 *
 * @param[in] x the point where log_prob and gradient were evaluated
 * @param[in] grad value of the gradient
 * @param[in] lp value of log probability
 * @return nothing
 */
void print_log_prob_grad_eval(std::vector<double> x, std::vector<double> grad, double lp) {
  std::cout << " * log_prob_grad() eval, x = ";
  print_vector(x);
  std::cout << ", lp = " << lp << ", grad = ";
  print_vector(grad);
  std::cout << "\n";
}


}  // namespace util
}  // namespace services
}  // namespace stan
#endif
