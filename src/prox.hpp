#ifndef LIB_PROX
#define LIB_PROX
#include <armadillo>
#include <iostream>
#include "feature.hpp"
using namespace arma;

vec soft_scale(vec x, double lambda);

void update_residuals(std::vector<feature *> features, residual *resid,
                      std::string loss_type);

vec fl_prox(feature *fused, residual *y, double lambda1, double lambda2,
            double t);
vec std_prox(feature *fused, residual *y, double lambda1, double t);

vec cat_prox(feature *fused, residual *y, double lambda1, double t);

vec intercept_prox(feature *x, residual *y, double t);

vec fit(feature *x, residual *y, double lambda1, double lambda2, double t);

mat new_fit(std::vector<feature *> features, residual *y, double lambda1,
            double lambda2, double t, vec to_fit);

mat get_fit_mat(std::vector<feature *> features);

bool linesearch(feature *x, residual *y, double lambda1, double lambda2,
                double t);

double fit_step(std::vector<feature *> features, residual *resid,
                double lambda1, double lambda2, double t, vec to_fit);

#endif /* PROX_H */