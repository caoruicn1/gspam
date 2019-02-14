
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

#include <math.h>
#include <iostream>
#include "prox.hpp"
#include "ryansdp.hpp"
#include "feature.hpp"
using namespace arma;
///@brief Update the residuals for a residual object, given a vector of features
///@param[in] features
/// A vector of pointers to features.
///@param[in] resid
/// Pointer to residual object
void update_residuals(std::vector<feature *> features, residual *resid,
                      std::string loss_type) {
  vec temp(resid->y_->n_rows);
  temp.fill(0);
  for (int i = 0; i < features.size(); i++) {
    temp = temp + *features.at(i)->fitted;
  };
  *resid->resid_old = *resid->resid;
  *resid->resid = grad(*resid->y_, temp, loss_type);
  vec resdiff = *resid->resid - *resid->resid_old;
  resid->resdiffnorm = norm(resdiff);
};

///@brief Solve the proximal problem for the fused lasso penalty.
///@param[in] fused
/// Pointer to feature object
///@param[in] y
/// Pointer to residual object
///@param[in] lambda1
/// Sparsity penalty
///@param[in] lambda2
/// Variational penalty
///@param[in] t
/// Step size
vec fl_prox(feature *fused, residual *y, double lambda1, double lambda2,
            double t) {
  /// Sort residuals and fitted values
  sort_by(y->resid, fused->buffer, fused->ord);
  sort_by(fused->fitted, fused->buffer, fused->ord);
  vec fused_fit = *fused->fitted;
  vec jresid = *fused->fitted - *y->resid * t;
  double *y_ptr = jresid.memptr();
  double *fitted_ptr = fused_fit.memptr();
  tf_dp(fused->fitted->n_rows, y_ptr, t * lambda2 * fused->fitted->n_rows,
        fitted_ptr);
  unsort_by(y->resid, fused->buffer, fused->ord);
  unsort_by(&fused_fit, fused->buffer, fused->ord);
  unsort_by(fused->fitted, fused->buffer, fused->ord);
  fused_fit = soft_scale(fused_fit, lambda1 * t * sqrt(fused->fitted->n_rows));
  return fused_fit;
};
///@brief Solve the proximal problem with no variational penalty (basis Q).
///@param[in] fused
/// Pointer to feature object
///@param[in] y
/// Pointer to residual object
///@param[in] lambda1
/// Sparsity penalty
///@param[in] t
/// Step size
vec std_prox(feature *fused, residual *y, double lambda1, double t) {
  vec jresid = *fused->fitted - *y->resid * t;  // y->resid->n_rows;
  jresid = (fused->Q * fused->Q.t()) * jresid;
  jresid = soft_scale(jresid, lambda1 * t * sqrt(fused->fitted->n_rows));
  return jresid;
};

///@brief Solve the proximal problem with no variational penalty for categorical
/// data.
///@param[in] fused
/// Pointer to feature object
///@param[in] y
/// Pointer to residual object
///@param[in] lambda1
/// Sparsity penalty
///@param[in] t
/// Step size
vec cat_prox(feature *fused, residual *y, double lambda1, double t) {
  vec jresid = *fused->fitted - *y->resid * t;
  vec *fitted = new vec(fused->fitted->n_rows);
  for (int i = 0; i < fused->fitted->n_rows; i++) {
    fused->buffer->at(fused->x->at(i)) =
        fused->buffer->at(fused->x->at(i)) + jresid.at(i);
  };
  for (int i = 0; i < fused->count->n_rows; i++) {
    fused->buffer->at(i) = fused->buffer->at(i) * (1.0 / fused->count->at(i));
  }
  for (int i = 0; i < fused->fitted->n_rows; i++) {
    fitted->at(i) = fused->buffer->at(fused->x->at(i));
  };
  fused->buffer->fill(0);
  *fitted = soft_scale(*fitted, lambda1 * t * sqrt(fused->fitted->n_rows));
  return *fitted;
};

vec intercept_prox(feature *x, residual *y, double t) {
  return *x->fitted - (sum(*y->resid) * t / y->resid->n_rows);
}


vec fit(feature *x, residual *y, double lambda1, double lambda2, double t) {
  if (x->prox_type == "fl") {
    return fl_prox(x, y, lambda1, lambda2, t);
  } else if (x->prox_type == "proj")
    return std_prox(x, y, lambda1, t);
  else if (x->prox_type == "cat")
    return cat_prox(x, y, lambda1, t);
  else if (x->prox_type == "intercept")
    return intercept_prox(x, y, t);
  else
    throw("Unsupported prox type.");
};

mat new_fit(std::vector<feature *> features, residual *y, double lambda1,
            double lambda2, double t, vec to_fit) {
  mat temp(y->resid->n_rows, features.size());
  temp.fill(0);
  for (int i = 0; i < features.size(); i++) {
    if (to_fit.at(i) == 1) {
      temp.col(i) = fit(features.at(i), y, lambda1, lambda2, t);
    }
  }
  return temp;
};

mat get_fit_mat(std::vector<feature *> features) {
  mat temp(features.at(0)->fitted->n_rows, features.size());
  temp.fill(0);
  for (int i = 0; i < features.size(); i++) {
    temp.col(i) = *features.at(i)->fitted;
  }
  return temp;
};

bool linesearch(residual *y, mat old_fit, mat new_fit, double t,
                std::string type) {
  vec ones_vec = ones<vec>(new_fit.n_cols);
  vec new_fit_t = (new_fit * ones_vec);
  vec old_fit_t = (old_fit * ones_vec);
  double actual = loss(*y->y_, new_fit_t, type);
  double approx = loss(*y->y_, old_fit_t, type);
  vec temp1 = grad(*y->y_, old_fit_t, y->loss_type);
  mat temp2 = (new_fit - old_fit);
  temp1 = temp2.t() * temp1;
  double grad_term = sum(temp1);
  double frob = norm(new_fit - old_fit, "fro");
  approx += (1.0 / (2.0 * t)) * pow(frob, 2);
  approx += grad_term;
  return (actual > approx);
};

///@brief Cycle through feature objects, updating the fitted values for each
///@param[in] features
/// Vector of pointers to feature objects
///@param[in] y
/// Pointer to residual object
///@param[in] lambda1
/// Sparsity penalty
///@param[in] lambda2
/// Variational penalty
///@param[in] t
/// Step size
double fit_step(std::vector<feature *> features, residual *resid,
                double lambda1, double lambda2, double t, vec to_fit) {
  mat old_fit = mat(resid->y_->n_rows, features.size());
  mat prox_fit = mat(resid->y_->n_rows, features.size());
  old_fit = get_fit_mat(features);
  prox_fit = new_fit(features, resid, lambda1, lambda2, t, to_fit);
  while (linesearch(resid, old_fit, prox_fit, t, resid->loss_type)) {
    t *= .8;
    prox_fit = new_fit(features, resid, lambda1, lambda2, t, to_fit);
  }
  // std::cout<<"RUNNING FIT w step stize : "<<t<<endl;
  int n = resid->resid->n_rows;
  for (int i = 0; i < features.size(); i++) {
    *features.at(i)->fitted = prox_fit.col(i);
  };
  // std::cout << "Fit of feature 9:"<< features.at(9)->fitted->at(0)<<endl;
  update_residuals(features, resid, resid->loss_type);
  return t;
};


///@brief Interpolate a fitted value for a single feature
///@param[in] feature
/// Feature object on which to interpolate
///@param[in] sort_point
/// X value of new point
///@return Fitted value estimate for new x value.
double interpolate(feature *x_sort, double sort_point) {
  int n = x_sort->x->n_rows;
  sort_by(x_sort->x, x_sort->buffer, x_sort->ord);
  sort_by(x_sort->fitted, x_sort->buffer, x_sort->ord);
  int l = 0;
  int u = n - 1;
  /// TODO: Should I extrapolate?
  if (sort_point < x_sort->x->at(l) || sort_point > x_sort->x->at(u))
    throw "New point outside allowable range, cannot interpolate.";
  while (u - l > 1) {
    if (sort_point < x_sort->x->at(u / 2)) {
      u = (l + u) / 2;
    } else if (sort_point > x_sort->x->at(u / 2)) {
      l = (l + u) / 2;
    } else if (sort_point == x_sort->x->at(u / 2)) {
      l = (l + u) / 2;
      u = (l + u) / 2;
    }
  }
  double fitted;
  if (u != l) {
    /// TODO: If there is a more compact way to do linear interpolation, do so.
    fitted = ((sort_point - x_sort->x->at(l)) /
              (x_sort->x->at(u) - x_sort->x->at(l))) *
                 (x_sort->fitted->at(u) - x_sort->fitted->at(l)) +
             x_sort->fitted->at(l);
  } else if (u == l)
    fitted = x_sort->fitted->at(l);
  unsort_by(x_sort->x, x_sort->buffer, x_sort->ord);
  unsort_by(x_sort->fitted, x_sort->buffer, x_sort->ord);
  return fitted;
};
///@brief Predict a new fitted value for a vector of covariates
///@param[in] features
/// Vector of pointers to feature objects
///@param[in] new_x
/// Vector of new x values to fit
///@return New fitted value for vector of x's
double predict(std::vector<feature *> features, vec new_x) {
  if (features.size() != new_x.n_rows)
    throw "New X vector does not have p elements.";
  vec temp = new_x;
  for (int i = 0; i < features.size(); i++) {
    temp.at(i) = interpolate(features.at(i), new_x.at(i));
  };
  double fitted = sum(temp);
  return fitted;
};


// Interpolation for new point
//' @title Interpolates a new value based on fitted value
//' @description Allows user to specify new point to predict on
//' @name interpolate
//' @param x covariate matrix
//' @param fitted matrix of fitted values
//' @param sort_point vector of new covariates to fit
//' @export
// [[Rcpp::export(name="interpolate_vec")]]
double interpolate_vec(arma::vec x, arma::vec fitted, double sort_point) {
  int n = x.n_rows;
  uvec ord = sort_index(x);
  vec *temp_ord = new vec(x.n_rows);
  sort_by(&x, temp_ord, ord);
  sort_by(&fitted, temp_ord, ord);
  int l = 0;
  int u = n - 1;
  if (sort_point < x.at(l)) {
    double new_fitted = fitted.at(l);
    unsort_by(&x, temp_ord, ord);
    unsort_by(&fitted, temp_ord, ord);
    return new_fitted;
  } else if (sort_point > x.at(u)) {
    double new_fitted = fitted.at(u);
    unsort_by(&x, temp_ord, ord);
    unsort_by(&fitted, temp_ord, ord);
    return new_fitted;
  }
  while (u - l > 1) {
    if (sort_point < x.at((l + u) / 2)) {
      u = (l + u) / 2;
    } else if (sort_point > x.at((l + u) / 2)) {
      l = (l + u) / 2;
    } else if (sort_point == x.at((l + u) / 2)) {
      l = (l + u) / 2;
      u = (l + u) / 2;
    }
  }
  double new_fitted;
  if (u != l) {
    /// TODO: If there is a more compact way to do linear interpolation, do so.
    new_fitted = ((sort_point - x.at(l)) / (x.at(u) - x.at(l))) *
                     (fitted.at(u) - fitted.at(l)) +
                 fitted.at(l);
  } else if (u == l)
    new_fitted = fitted.at(l);
  unsort_by(&x, temp_ord, ord);
  unsort_by(&fitted, temp_ord, ord);
  return new_fitted;
};
