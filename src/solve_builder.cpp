
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#include "solve_builder.hpp"

#define EPS .01
#define MAXIT 1000

arma::mat single_solver(std::vector<feature *> features, residual *y,
                        double lambda1, double lambda2, vec *ever_active) {
  int i = 0;
  double t = 2;
  y->resdiffnorm = 1000;
  vec first_step(features.size());
  first_step.fill(1);
  vec ever_active_old(features.size());
  ever_active_old.fill(0);
  do {
    ever_active_old = *ever_active;
    do {
      t = fit_step(features, y, lambda1, lambda2, t, *ever_active);
      i++;
    } while (y->resdiffnorm > EPS & i < MAXIT);
    t = fit_step(features, y, lambda1, lambda2, t, first_step);
    for (int i = 0; i < features.size(); i++) {
      if (norm(*features.at(i)->fitted) > 0) {
        ever_active->at(i) = 1;
      }
    }
  } while (!approx_equal(*ever_active, ever_active_old, "absdiff", 0.002));

  arma::mat fitted = mat(y->resid->n_rows, features.size(), fill::zeros);
  for (int i = 0; i < features.size(); i++) {
    fitted.col(i) = *features.at(i)->fitted;
  };
  return fitted;
};

field<arma::mat> path_solver(std::vector<feature *> features, residual *y,
                             vec lambda1, vec lambda2) {
  vec *ever_active = new vec(features.size());
  ever_active->fill(0);
  field<arma::mat> fitted(lambda1.n_rows);
  if (lambda1.n_rows == lambda2.n_rows) {
    for (int i = lambda1.n_rows - 1; i >= 0; i--) {
      fitted(i) =
          single_solver(features, y, lambda1.at(i), lambda2.at(i), ever_active);
    }
  } else
    throw "Lambda Vectors are of invalid length (must match each other)";
  return fitted;
};

std::vector<feature *> feature_builder(arma::mat data,
                                       std::vector<std::string> prox_type,
                                       residual *y) {
  std::vector<feature *> features;
  vec temp;
  feature *temp1 = new feature("intercept", y);
  features.push_back(temp1);
  for (int i = 0; i < prox_type.size(); i++) {
    temp = data.col(i);
    feature *temp1 = new feature(prox_type.at(i), &temp);
    features.push_back(temp1);
  }
  return features;
};

arma::mat gspam_solver(arma::mat data, arma::vec y,
                       std::vector<std::string> prox_type,
                       std::string loss_type, double lambda1, double lambda2) {
  residual *y_resid = new residual(&y, loss_type);
  std::vector<feature *> features = feature_builder(data, prox_type, y_resid);
  update_residuals(features, y_resid, y_resid->loss_type);
  vec *ever_active = new vec(features.size());
  ever_active->fill(0);
  arma::mat fitted =
      single_solver(features, y_resid, lambda1, lambda2, ever_active);
  for (int i = 0; i < features.size(); i++) {
    delete (features.at(i));
  }
  return fitted;
};

field<arma::mat> gspam_path_solver(mat data, vec y,
                                   std::vector<std::string> prox_type,
                                   std::string loss_type, vec lambda1,
                                   vec lambda2) {
  residual *y_resid = new residual(&y, loss_type);
  std::vector<feature *> features = feature_builder(data, prox_type, y_resid);
  update_residuals(features, y_resid, y_resid->loss_type);
  field<arma::mat> fits = path_solver(features, y_resid, lambda1, lambda2);
  for (int i = 0; i < features.size(); i++) {
    delete (features.at(i));
  }
  return fits;
};

arma::vec lambda_path(std::vector<feature *> features, residual *y,
                      double alpha) {
  double u;
  double l = 0;
  u = pow(norm(*y->y_), 2);
  mat test_fit = mat(y->y_->n_rows, features.size());
  test_fit.fill(0.0);
  vec ever_active(features.size());
  ever_active.fill(1);
  while (u - l > EPS) {
    test_fit = new_fit(features, y, (u + l) / 2.0, alpha * (u + l) / 2.0,
                       1.0 / features.size(), ever_active);
    test_fit.shed_col(0);
    if (norm(test_fit, "fro") == 0) {
      u = (u + l) / 2.0;
    } else
      l = (u + l) / 2.0;
  }
  vec lambda_vec;
  lambda_vec = exp(linspace<vec>(log(.01 * u), log(u), 100));
  return lambda_vec;
};
