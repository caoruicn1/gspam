#ifndef UTIL_H
#define UTIL_H
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#include <math.h>
#include <iostream>
using namespace arma;

vec soft_scale(vec x, double lambda);


double quad_loss(vec y, vec theta);

double log_loss(vec y, vec theta);

vec quad_grad(vec y, vec theta);

vec log_grad(vec y, vec theta);

double loss(vec y, vec theta, std::string type);

vec grad(vec y, vec theta, std::string type);
#endif /* UTIL_H */