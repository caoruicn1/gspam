% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/RcppExports.R
\name{get_lambdas}
\alias{get_lambdas}
\title{Lambda path retrieval for given data and outcome}
\usage{
get_lambdas(data, y, prox_type, loss_type, alpha)
}
\arguments{
\item{data}{n by p matrix of inputs}

\item{y}{response column vector}

\item{prox_type}{type of prox to use}

\item{loss_type}{type of loss to use}

\item{alpha}{scalar value lambda2=alpha*lambda1}
}
\description{
Lambda path retrieval for given data and outcome
}
