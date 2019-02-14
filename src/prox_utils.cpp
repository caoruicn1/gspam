// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"
// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#include <math.h>
#include <iostream>
#include "prox_utils.hpp"
using namespace arma;
// Utils for prox functions

///@brief Soft scale a vector x by a value lambda
///@param[in] x
/// Vector to scale
///@param[in] lambda
/// Value by which to scale
///@return Scaled vector
vec soft_scale(vec x, double lambda) {
  double temp = (1 - (lambda) / norm(x));
  // std::cout<<"lambda: "<<lambda<<endl;
  // std::cout<<"x: "<<x<<endl;
  // std::cout<<"temp: "<<temp<<endl;
  if (temp > 0) {
    x = temp * x;
  } else {
    x = 0.0 * x;
  }
  return x;
};

/// NEW LOSSES GO HERE

double quadloss(vec y, vec theta) {
  double temp = norm(y - theta, 2);
  return (temp * temp);
};

double log_loss(vec y, vec theta) {
  vec t = exp(theta);
  vec t1 = t / (1 + t);
  double temp = dot(y, t1);
  return temp + dot((1 - y), (1 / (1 + exp(theta))));
};

/// NEW GRADIENTS GO HERE (and in header)

vec quad_grad(vec y, vec theta) { return 2 * (theta - y); };

vec log_grad(vec y, vec theta) { return y - (exp(theta) / (1 + exp(theta))); };

///########

/// PUT TYPE IN IF STATEMENT HERE IN BOTH LOSS AND GRAD FUNCTIONS

double loss(vec y, vec theta, std::string type) {
  if (type == "quad") {
    return quad_loss(y, theta);
  } else if (type == "log") {
    return log_loss(y, theta);
  } else{
    throw std::invalid_argument( "Bad loss type" );
  }
};

vec grad(vec y, vec theta, std::string type) {
  if (type == "quad") {
    return quad_grad(y, theta);
  } else if (type == "log") {
    return log_grad(y, theta);
  } else{
    throw std::invalid_argument( "Bad loss type" );
  }
};