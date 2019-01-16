#ifndef UTIL_H
#define UTIL_H
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#include <iostream>
#include <math.h>
using namespace arma;

vec soft_scale(vec x, double lambda);


double quadloss(vec y, vec theta) ;

double logloss(vec y, vec theta);
  
vec quadgrad(vec y, vec theta);  

vec loggrad(vec y, vec theta);

double loss(vec y, vec theta, std::string type);

vec grad(vec y, vec theta, std::string type);
#endif /* UTIL_H */