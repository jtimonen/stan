#ifndef STAN_MCMC_HMC_HAMILTONIANS_BASE_HAMILTONIAN_HPP
#define STAN_MCMC_HMC_HAMILTONIANS_BASE_HAMILTONIAN_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/model/gradient.hpp>
#include <stan/model/log_prob_propto.hpp>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace stan {
namespace mcmc {

template <class Model, class Point, class BaseRNG>
class base_hamiltonian {
 public:
  explicit base_hamiltonian(const Model& model) : model_(model) {}

  ~base_hamiltonian() {}

  typedef Point PointType;

  virtual double T(Point& z) = 0;

  double V(Point& z) { return z.V; }

  virtual double tau(Point& z) = 0;

  virtual double phi(Point& z) = 0;

  double H(Point& z) { return T(z) + V(z); }

  // The time derivative of the virial, G = \sum_{d = 1}^{D} q^{d} p_{d}.
  virtual double dG_dt(Point& z, callbacks::logger& logger) = 0;

  // tau = 0.5 p_{i} p_{j} Lambda^{ij} (q)
  virtual Eigen::VectorXd dtau_dq(Point& z, callbacks::logger& logger) = 0;

  virtual Eigen::VectorXd dtau_dp(Point& z) = 0;

  // phi = 0.5 * log | Lambda (q) | + V(q)
  virtual Eigen::VectorXd dphi_dq(Point& z, callbacks::logger& logger) = 0;

  virtual void sample_p(Point& z, BaseRNG& rng) = 0;

  void init(Point& z, callbacks::logger& logger) {
    this->update_potential_gradient(z, logger);
  }

  void update_potential(Point& z, callbacks::logger& logger) {
    try {
      z.V = -stan::model::log_prob_propto<true>(model_, z.q);
    } catch (const std::exception& e) {
      this->write_error_msg_(e, logger, "update_potential");
      z.V = std::numeric_limits<double>::infinity();
    }
  }

  void update_potential_gradient(Point& z, callbacks::logger& logger) {
    try {
      stan::model::gradient(model_, z.q, z.V, z.g, logger);
      z.V = -z.V;
    } catch (const std::exception& e) {
      this->write_error_msg_(e, logger, "update_potential_gradient");
      z.V = std::numeric_limits<double>::infinity();
    }
    z.g = -z.g;
  }

  void update_metric(Point& z, callbacks::logger& logger) {}

  void update_metric_gradient(Point& z, callbacks::logger& logger) {}

  void update_gradients(Point& z, callbacks::logger& logger) {
    update_potential_gradient(z, logger);
  }

 protected:
  const Model& model_;

  void write_error_msg_(const std::exception& e, callbacks::logger& logger,
       std::string function_name) {
    logger.error("Current Metropolis proposal is about to be rejected ");
    logger.error("due to the following issue (in '" + function_name + "'): ");
    logger.error(e.what());
    logger.error(
      "If this occurs often your model may be "
      "ill-conditioned or misspecified.");
    logger.error("");
  }
};

}  // namespace mcmc
}  // namespace stan

#endif
