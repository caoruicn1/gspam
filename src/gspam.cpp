
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#include <iostream>
#include <string>
#include "solve_builder.hpp"

// Generalized Sparse Additive Model Solver
//' @title Default Generalized Sparse Additive Model Solver with single
//' Prespecified variational penalty and lambda's
//' @description Fit model as specified by user for a single variational penalty
//' (matrix of X-values).
//' @name gspam_c
//' @param data n by p matrix of inputs
//' @param y response column vector
//' @param type of prox to use
//' @param sparsity penalty
//' @param variational penalty
//' @export
// [[Rcpp::export(name="gspam_c")]]
Rcpp::List gspam_c(arma::mat data, arma::vec y,
                   std::vector<std::string> prox_type, std::string loss_type,
                   double lambda1 = 0.0, double lambda2 = 0.0) {
  arma::mat gsp_fitted =
      gspam_solver(data, y, prox_type, loss_type, lambda1, lambda2);
  return Rcpp::List::create(Rcpp::Named("data", data),
                            Rcpp::Named("fitted", gsp_fitted),
                            Rcpp::Named("y", y));
};

// Generalized Sparse Additive Model Solver for vector of values
//' @title Default Generalized Sparse Additive Model Solver with vector of
//' prespecified variational penalties and sparsity penalties
//' @description Fit model as specified by user for a single variational penalty
//' (matrix of X-values).
//' @name gspam_c_vec
//' @param data n by p matrix of inputs
//' @param y response column vector
//' @param type of prox to use
//' @param sparsity penalties
//' @param variational penalties
//' @export
// [[Rcpp::export(name="gspam_c_vec")]]
Rcpp::List gspam_c_vec(arma::mat data, arma::vec y,
                       std::vector<std::string> prox_type,
                       std::string loss_type, arma::vec lambda1,
                       arma::vec lambda2) {
  if (lambda1.n_rows != lambda2.n_rows) {
    throw "Variational and sparsity vectors must be of same size.";
  }
  arma::field<mat> gsp_fitted =
      gspam_path_solver(data, y, prox_type, loss_type, lambda1, lambda2);
  return Rcpp::List::create(
      Rcpp::Named("data", data), Rcpp::Named("fitted", gsp_fitted),
      Rcpp::Named("y", y), Rcpp::Named("lambda1", lambda1),
      Rcpp::Named("lambda2", lambda2));
};

// Interpolation for new point
//' @title Interpolates a new value based on fitted value
//' @description Allows user to specify new point to predict on
//' @name interpolate
//' @param x covariate matrix
//' @param fitted matrix of fitted values
//' @param sort_point vector of new covariates to fit
//' @export
// [[Rcpp::export(name="interpolate")]]
double interpolate(arma::mat x, arma::mat fitted, arma::vec sort_point) {
  double new_fitted = fitted.at(0, 0);
  for (int i = 0; i < x.n_cols; i++) {
    new_fitted +=
        interpolate_vec(x.col(i), fitted.col(i + 1), sort_point.at(i));
  }
  return new_fitted;
}

// Generalized Sparse Additive Model Solver for vector of values
//' @title Default Generalized Sparse Additive Model Solver that calculates
//' lambda path based on mixture value.
//' @description Fit model as specified by user for calculated range of lambda
//' values.
//' @name gspam_full
//' @param data n by p matrix of inputs
//' @param y response column vector
//' @param type of prox to use
//' @param scalar value lambda2=alpha*lambda1
//' @export
// [[Rcpp::export(name="gspam_full")]]
Rcpp::List gspam_full(arma::mat data, arma::vec y,
                      std::vector<std::string> prox_type, std::string loss_type,
                      double alpha) {
  residual *y_resid = new residual(&y, loss_type);
  std::vector<feature *> features = feature_builder(data, prox_type, y_resid);
  update_residuals(features, y_resid, loss_type);
  vec lambdapath = lambda_path(features, y_resid, alpha);
  for (int i = 0; i < features.size(); i++) {
    delete (features.at(i));
  }
  return gspam_c_vec(data, y, prox_type, loss_type, lambdapath,
                     alpha * lambdapath);
};


// Get lambda path from data
//' @title Lambda path retrieval for given data and outcome
//' @name get_lambdas
//' @param data n by p matrix of inputs
//' @param y response column vector
//' @param type of prox to use
//' @param scalar value lambda2=alpha*lambda1
//' @export
// [[Rcpp::export(name="get_lambdas")]]
Rcpp::List get_lambdas(arma::mat data, arma::vec y,
                       std::vector<std::string> prox_type,
                       std::string loss_type, double alpha) {
  residual *y_resid = new residual(&y, loss_type);
  std::vector<feature *> features = feature_builder(data, prox_type, y_resid);
  update_residuals(features, y_resid, loss_type);
  vec lambda1 = lambda_path(features, y_resid, alpha);
  vec lambda2 = alpha * lambda1;
  for (int i = 0; i < features.size(); i++) {
    delete (features.at(i));
  }
  return Rcpp::List::create(Rcpp::Named("lambda1", lambda1),
                            Rcpp::Named("lambda2", lambda2));
};