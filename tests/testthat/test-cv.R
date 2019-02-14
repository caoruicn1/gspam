context("test-cv")
library(gspam)
test_that("quad", {
  # TESTS
  set.seed(1408)
  # FL prox, Quad loss
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
test_that("log", {
  # TESTS
  set.seed(1408)
  # FL prox, log loss
  # Set Params
  # Number of obs
  n <- 100
  # Number of features
  p <- 100
  # Number of signal features
  s <- 5
  # Ratio of shape tuning param to sparsity tuning param
  alpha <- 0.5
  # X is random normal
  X <- matrix(rnorm(n*p),ncol=p)
  beta <- c(rep(1,s),rep(0,p-s))
  sig <- sign(X)
  y <- rbinom(n,size=1,prob=expit(sig %*% beta))
  prox_type <- rep("fl",p)
  start.time <- Sys.time()
  results <- gspam_full(X,y,prox_type,"log",0.5)
  end.time <- Sys.time()
  end.time-start.time
})
test_that("catlog", {
  # TESTS
  set.seed(1408)
  # FL prox, log loss
  # Set Params
  # Number of obs
  n <- 1000
  # Number of features
  p <- 100
  # Number of signal features
  s <- 5
  # Ratio of shape tuning param to sparsity tuning param
  alpha <- 0.5
  cats <- c("yes", "no","maybe")
  cat_data <- sample(cats, n, replace =T)
  cats <-rownames(table(cat_data))
  numdata<-match(cat_data,cats)-1
  # X is random normal
  X <- matrix(rnorm(n*p),ncol=p)
  beta <- c(rep(1,s),rep(0,p-s))
  sig <- sign(X)
  y <- rbinom(n,size=1,prob=expit(sig %*% beta+(numdata== 1)-2*(numdata == 2)))
  prox_type <- c("cat",rep("fl",p))
  start.time <- Sys.time()
  results <- gspam_full(cbind(numdata,X),y,prox_type,"log",0.5)
  end.time <- Sys.time()
  end.time-start.time
})
