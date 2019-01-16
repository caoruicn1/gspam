
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
#include "prox.hpp"
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

arma::mat single_solver(std::vector<feature *> features, residual *y,
                        double lambda1, double lambda2);
field<arma::mat> path_solver(std::vector<feature *> features, residual *y,
                             vec lambda1, vec lambda2);
std::vector<feature *> feature_builder(arma::mat data,
                                       std::vector<std::string> prox_type, residual *y);

field<arma::mat> gspam_path_solver(mat data, vec y,
                                   std::vector<std::string> prox_type,std::string loss_type,
                                   vec lambda1, vec lambda2);
arma::mat gspam_solver(arma::mat data, arma::vec y,
                       std::vector<std::string> prox_type, std::string loss_type, double lambda1,
                       double lambda2);
arma::vec lambda_path(std::vector<feature *> features, residual *y,
                      double alpha);