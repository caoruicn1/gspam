% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gspam.R
\name{plot.gspam}
\alias{plot.gspam}
\alias{plot.gspam.cv}
\title{Plot the sparsity vs. lambda for a fit of gspam}
\usage{
\method{plot}{gspam}(x)

\method{plot}{gspam.cv}(x)
}
\arguments{
\item{x}{fitted gspam.cv object}

\item{fitted}{gspam object}
}
\description{
Plots the number of active features as log(lambda1) varies

Plots the mean of specified loss vs log(lambda1) for a fitted gspam.cv object
}