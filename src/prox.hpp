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

double fit_step(std::vector<feature *> features, residual *resid,
                double lambda1, double lambda2, double t, vec to_fit);

double interpolate(feature *x_sort, double sort_point);

double predict(std::vector<feature *> features, vec new_x);

vec cat_prox(feature *fused, residual *y, double lambda1, double t);

double interpolate_vec(vec x, vec fitted, double sort_point);


bool linesearch(feature *x, residual *y, double lambda1, double lambda2,
                double t);


mat new_fit(std::vector<feature *> features, residual *y, double lambda1,
            double lambda2, double t, vec to_fit);

mat get_fit_mat(std::vector<feature *> features);

bool l2_linesearch(residual *y, mat old_fit, mat new_fit, double t);

vec fit(feature *x, residual *y, double lambda1, double lambda2, double t);

double interpolate(feature *x_sort, double sort_point);


double predict(std::vector<feature *> features, vec new_x);

double interpolate_vec(vec x, vec fitted, double sort_point);

vec soft_scale(vec x, double lambda);


double quadloss(vec y, vec theta);


double logloss(vec y, vec theta);

vec quadgrad(vec y, vec theta);

vec loggrad(vec y, vec theta);

double loss(vec y, vec theta, std::string type);

vec grad(vec y, vec theta, std::string type);

#endif /* PROX_H */