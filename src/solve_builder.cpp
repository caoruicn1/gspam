
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
                        double lambda1, double lambda2) {
  int i = 0;
  double t=2;
  y->resdiffnorm = 1000;
  //cout<< "(l1,l2)= : "<<lambda1<<" , "<<lambda2<<endl;
  while (y->resdiffnorm > EPS & i < MAXIT) {
    t = fit_step(features, y, lambda1, lambda2,t);
    i++;
  }
  //cout<< "converged in "<< i << " steps."<<endl;
  arma::mat fitted = mat(y->resid->n_rows, features.size(), fill::zeros);
  for (int i = 0; i < features.size(); i++) {
    fitted.col(i) = *features.at(i)->fitted;
  };
  return fitted;
};

field<arma::mat> path_solver(std::vector<feature *> features, residual *y,
                             vec lambda1, vec lambda2) {
  field<arma::mat> fitted(lambda1.n_rows);
  if (lambda1.n_rows == lambda2.n_rows) {
    for (int i = lambda1.n_rows-1; i >=0; i--) {
      fitted(i) = single_solver(features, y, lambda1.at(i), lambda2.at(i));
    }
  } else
    throw "Lambda Vectors are of invalid length (must match each other)";
  return fitted;
};

std::vector<feature *> feature_builder(arma::mat data,
                                       std::vector<std::string> prox_type, residual *y) {
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
                       std::vector<std::string> prox_type,std::string loss_type, double lambda1,
                       double lambda2) {
  residual *y_resid = new residual(&y,loss_type);
  std::vector<feature *> features = feature_builder(data, prox_type,y_resid);
  update_residuals(features, y_resid,y_resid->loss_type);
  arma::mat fitted = single_solver(features, y_resid, lambda1, lambda2);
  return fitted;
};

field<arma::mat> gspam_path_solver(mat data, vec y,
                                   std::vector<std::string> prox_type,std::string loss_type,
                                   vec lambda1, vec lambda2) {
  residual *y_resid = new residual(&y,loss_type);
  std::vector<feature *> features = feature_builder(data, prox_type,y_resid);
  update_residuals(features, y_resid,y_resid->loss_type);
  field<arma::mat> fits = path_solver(features, y_resid, lambda1, lambda2);
  return fits;
};

arma::vec lambda_path(std::vector<feature *> features, residual *y,
                      double alpha) {
  double u;
  double l = 0;
  u = pow(norm(*y->y_), 2);
  mat test_fit = mat(y->y_->n_rows, features.size());
  test_fit.fill(0.0);
  while (u - l > EPS) {
    test_fit = new_fit(features, y, (u + l) / 2.0, alpha * (u + l) / 2.0,
                       1.0 / features.size());
    test_fit.shed_col(0);
    if (norm(test_fit, "fro") == 0) {
      u = (u + l) / 2.0;
    } else
      l = (u + l) / 2.0;
  }
  vec lambda_vec;
  lambda_vec = exp(linspace<vec>(log(.01*u),  log(u), 100));
  return lambda_vec;
};

//
//
//
// // Generalized Sparse Additive Model Solver for categorical vector of
// features
// //' @title Default Generalized Sparse Additive Model Solver with single
// Prespecified variational penalty and lambda's
// //' @description Fit model as specified by user for a single variational
// penalty (matrix of X-values).
// //' @name gspam
// //' @param vector of numeric categories
// //' @param y response column vector
// //' @param type of prox to use
// //' @param sparsity penalty
// //' @param variational penalty
// //' @export
// // [[Rcpp::export(name="gspam_cat_solver")]]
// arma::mat gspam_cat_solver(arma::mat data, arma::vec y, arma::mat counts,
// std::string prox_type, double lambda1){
//   std::vector<feature *> features;
//   vec temp;
//   vec coltemp;
//   for (int i = 0; i < data.n_cols; i++) {
//     temp = data.col(i);
//     coltemp=counts.col(i);
//     feature *temp1 = new feature(prox_type, &temp,&coltemp);
//     features.push_back(temp1);
//   };
//   residual *test_y = new residual(&y);
//   int i=0;
//   while (test_y->resdiffnorm > EPS & i < MAXIT) {
//     test_y->resid->print();
//     fit_step(features, test_y, lambda1,0,1.0/features.size());
//     i++;
//   }
//   arma::mat fitted=mat(data.n_rows, data.n_cols, fill::zeros);
//   cout<<data.n_cols<<endl;
//   for (int i = 0; i < data.n_cols; i++) {
//     fitted.col(i)=*features.at(i)->fitted;
//   };
//   return fitted;
// }
