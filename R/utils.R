# Helper Functions for R

#' #' @title expit
#' #' @description expit function
#' #' @name expit
#' #' @param vector or scalar to use
#' #' @export
expit <- function(x){
  return(exp(x)/(1+exp(x)))
}

#' #' @title logit
#' #' @description logit function
#' #' @name logit
#' #' @param vector or scalar to use
#' #' @export
logit <- function(x){
  return(log(x/(1-x)))
}