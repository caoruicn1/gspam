context("test-cv")
library(gspam)
test_that("cv valgrind", {
  # TESTS
  set.seed(1408)
  # FL prox, Quad loss
  # Num trials
  N <- 10
  # Set Params
  n <- 100
  p <- 100
  s <- 5
  alpha <- 0.5
  X <- matrix(rnorm(n*p),ncol=p)
  beta <- c(rep(1,s),rep(0,p-s))
  sig <- sign(X)
  y <- rnorm(n)+sig %*% beta
  prox_type <- rep("fl",p)
  start.time <- Sys.time()
  results <- gspam.cv(X,y,prox_type,"quad",0.5)
  end.time <- Sys.time()
  end.time-start.time
})
test_that("logit", {
  # TESTS
  set.seed(1408)
  # FL prox, Quad loss
  # Num trials
  N <- 10
  # Set Params
  n <- 100
  p <- 100
  s <- 5
  alpha <- 0.5
  X <- matrix(rnorm(n*p),ncol=p)
  beta <- c(rep(1,s),rep(0,p-s))
  sig <- sign(X)
  y <- rnorm(n)+sig %*% beta
  prox_type <- rep("fl",p)
  start.time <- Sys.time()
  results <- gspam_full(X,y,prox_type,"quad",0.5)
  end.time <- Sys.time()
  end.time-start.time
})
