#include "feature.hpp"
#include <RcppArmadillo.h>
#include <iostream>
using namespace arma;

void sort_by(vec *orig, vec *buffer, uvec ord) {
  for (int i = 0; i < orig->n_rows; i++) {
    buffer->at(i) = orig->at(ord.at(i));
  };
  *orig = *buffer;
  return;
};

void unsort_by(vec *orig, vec *buffer, uvec ord) {
  for (int i = 0; i < orig->n_rows; i++) {
    buffer->at(ord.at(i)) = orig->at(i);
  };
  *orig = *buffer;
  return;
};
