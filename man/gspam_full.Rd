% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{gspam_full}
\alias{gspam_full}
\title{Default Generalized Sparse Additive Model Solver that calculates
lambda path based on mixture value.}
\usage{
gspam_full(data, y, prox_type, loss_type, alpha)
}
\arguments{
\item{data}{n by p matrix of inputs}

\item{y}{response column vector}

\item{prox_type}{type of prox to use}

\item{loss_type}{type of loss to use}

\item{alpha}{scalar value lambda2=alpha*lambda1}
}
\description{
Fit model as specified by user for calculated range of lambda
values.
}
