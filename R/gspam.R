# Generalized Sparse Additive Model Solver with crossvalidation
#' @title Crossvalidation for gspam
#' @description Fits over generated lambda path for specified prox and loss.
#' @name gspam.cv
#' @param x covariate matrix/data frame
#' @param y response vector, if the loss type is "quad" then this is a vector of real-values.
#' If the loss type is "log" then this is a vector of binary values.
#' @param prox_type vector of prox types to use, or a single prox type
#' @param loss_type type of loss to use
#' @param alpha mixture of lambdas (lambda2= alpha*lambda1)
#' @param k number of folds, default is 10
#' @export
gspam <- function(x, y, prox_type, loss_type, alpha = 0.5) {
  if (length(prox_type) < ncol(x)) {
    prox_type <- rep(prox_type[1], ncol(x))
  }
  temp_list <- prep_df(x, prox_type)
  x_mat <- as.matrix(temp_list[[1]])
  mappings  <- temp_list[[2]]
  prox_type <- temp_list[[3]]
  results <- gspam:::gspam_full(x_mat, y, prox_type, loss_type, alpha)
  results$data <- x
  if(is.data.frame(x)){
  for (i in 1:100) {
    colnames(results$fitted[[i]]) <- c('intercept', names(x))
  }
  }
  class(results) <- c('gspam')
  return(results)
}


# Generalized Sparse Additive Model Solver with crossvalidation
#' @title Crossvalidation for gspam
#' @description Splits data into train and test sets, tests over lambda path for entire data.
#' @name gspam.cv
#' @param data covariate matrix/data frame
#' @param y response vector
#' @param prox_type vector of prox types to use
#' @param loss_type type of loss to use
#' @param alpha mixture of lambdas (lambda2= alpha*lambda1)
#' @param k number of folds, default is 10
#' @export
gspam.cv <- function(x,
                     y,
                     prox_type,
                     loss_type,
                     alpha = 0.5,
                     k = 10,
                     folds = NULL) {
  if (length(prox_type) < ncol(x)) {
    prox_type <- rep(prox_type[1], ncol(x))
  }
  temp_list <- prep_df(x, prox_type)
  data <- as.matrix(temp_list[[1]])
  mappings  <- temp_list[[2]]
  prox_type <- temp_list[[3]]
  # Get lambda vector to do cv over frow full data
  lambdas <- get_lambdas(data, y, prox_type, loss_type, alpha)
  lambda1 <- lambdas$lambda1
  lambda2 <- lambdas$lambda2
  if(is.null(folds)){
  # Randomly shuffle the data
  randomizer <- sample(nrow(data))
  rand_X <- data[randomizer, ]
  rand_y <- y[randomizer]
  # Create k equally size folds
  folds <- cut(seq(1, nrow(data)), breaks = k, labels = FALSE)
  }
  else{
    folds <- as.integer(folds)
    k <- length(unique(folds))
  }
  # Perform k fold cross validation
  mselist <- matrix(rep(0, k * 100), ncol = k)
  for (i in 1:k) {
    testIndexes <- which(folds == i, arr.ind = TRUE)
    testX <- rand_X[testIndexes,]
    testy <- rand_y[testIndexes]
    trainX <- rand_X[-testIndexes,]
    trainy <- rand_y[-testIndexes]
    result <-
      gspam_c_vec(trainX, trainy, prox_type, loss_type, lambda1, lambda2)
    if (is.nan(result$fitted[[20]][5, 2])) {
      return(c(k,folds,result$fitted))
    }
    test_fits <- c()
    for (j in 1:100) {
      fits <- rep(result$fitted[[j]][1, 1], nrow(testX))
      for (l in 1:ncol(testX)) {
        fits <-
          fits + approx(x = trainX[, l],
                        y = result$fitted[[j]][, l + 1],
                        testX[, l],
                        rule = 2)$y
      }
      test_fits[[j]] <- fits
    }
    mse <- c()
    for (j in 1:100) {
      mse[[j]] <- loss(testy, test_fits[[j]], loss_type) / length(testy)
    }
    mselist[, i] <- mse
  }
  fitted <- gspam_full(data, y, prox_type, loss_type, alpha = 0.5)
  best_lambda1 <- lambda1[which.min(rowMeans(mselist))]
  best_fit <- fitted$fitted[[which(lambda1 == best_lambda1)]]
  sds <- apply(mselist, 1, sd) / sqrt(k)
  lower <- rowMeans(mselist) - 1.96 * sds
  upper <- rowMeans(mselist) + 1.96 * sds
  results <-
    list(
      'data' = data,
      "errors" = rowMeans(mselist),
      "lambda1" = lambda1,
      "best_lambda1" = best_lambda1,
      lowermse = lower,
      uppermse = upper,
      full = mselist,
      'best_fit' = best_fit
    )
  class(results) <- c("gspam.cv")
  return(results)
}

## Auxilary function to get numeric versions of categorical variables and their reverse mapping
prep_df <- function(df, prox_type) {
  mappings <- c()
  t <- 1
  for (i in 1:ncol(df)) {
    if (class(df[, i]) %in% c("factor", "character")) {
      ints <- as.integer(df[, i])
      temp_df <- cbind(ints, as.data.frame(df[, i]))
      temp_df <- unique(temp_df[order(temp_df[, 2]), ])
      names(temp_df) <- c("Number", "Category")
      mappings[[t]] <- temp_df
      prox_type[i] <- 'cat'
      df[, i] <- ints
      t <- t + 1
    }
  }
  return(list(df, mappings, prox_type))
}

#' @title Predict Function
#' @description Predict new y values from a data frame or matrix of covariates with the same number of columns as the original
#' @name predict
#' @param data object of class gspam
#' @param new_points feature matrix or data frame.
#' @param p scalar for lambda position to use (1 is the lowest lambda pair on the path)
#' @export
predict.gspam <- function(data, new_points, p) {
  fitted <-
    matrix(rep(0, nrow(new_points) * ncol(new_points)), ncol = ncol(new_points))
  old_fitted <- data$fitted[[p]]
  old_x <- data$data
  if (ncol(new_points) != (ncol(old_x))) {
    throw("New x object does not have same number of columns as original fitted object.")
  }
  for (i in 1:ncol(old_x)) {
    if (class(new_points[, i]) %in% c("factor", "character")) {
      fitted[, i] <- old_fitted[match(new_points[, i], old_x[, i]), (i + 1)]
    }
    fitted[, i] <-
      approx(
        x = old_x[, i],
        y = old_fitted[, (i + 1)],
        xout = new_points[, i],
        rule = 2
      )$y
  }
  predictions <- rowSums(fitted) + old_fitted[1, 1]
  return(predictions)
}

#' @title Predict Function for cv object
#' @description Predict new y values from a data frame or matrix of covariates with the same number of columns as the original.
#'  Uses the fit with the lowest MSE from cv
#' @name predict
#' @param data object of class gspam.cv
#' @param new_points feature matrix or data frame.
#' @param p scalar for lambda position to use (1 is the lowest lambda pair on the path)
#' @export
predict.gspam.cv <- function(data, new_points) {
  fitted <-
    matrix(rep(0, nrow(new_points) * ncol(new_points)), ncol = ncol(new_points))
  old_fitted <- data$best_fit
  old_x <- data$data
  if (ncol(new_points) != (ncol(old_x))) {
    throw("New x object does not have same number of columns as original fitted object.")
  }
  for (i in 1:ncol(old_x)) {
    if (class(new_points[, i]) %in% c("factor", "character")) {
      fitted[, i] <- old_fitted[match(new_points[, i], old_x[, i]), (i + 1)]
    }
    approx(x = old_x[, i], y = old_fitted[, (i + 1)], xout = new_points[, i])$y
    fitted[, i] <-
      approx(
        x = old_x[, i],
        y = old_fitted[, (i + 1)],
        xout = new_points[, i],
        rule = 2
      )$y
  }
  predictions <- rowSums(fitted) + old_fitted[1, 1]
  return(predictions)
}

#' @title Plot the sparsity vs. lambda for a fit of gspam
#' @description Plots the number of active features as log(lambda1) varies
#' @name plot.gspam
#' @param fitted gspam object
#' @export
plot.gspam <- function(x) {
  active_list <- rep(0, 100)
  for (i in 1:100) {
    curr_active <- 0
    temp <- x$fitted[[i]]
    for (j in 2:ncol(temp)) {
      if (sum(temp[, j] ^ 2) != 0) {
        curr_active <- curr_active + 1
      }
    }
    active_list[i] <- curr_active
  }
  plot(
    log(x$lambda1),
    active_list,
    xlab = "Log Lambda",
    ylab = "Number of Active Features",
    main = "Active Features Plot"
  )
}

#' @title Plot the mean loss vs log lambda
#' @description Plots the mean of specified loss vs log(lambda1) for a fitted gspam.cv object
#' @name plot.gspam
#' @param x fitted gspam.cv object
#' @export
plot.gspam.cv <- function(x) {
  plot(
    log(x$lambda1),
    x$error,
    ylim = range(c(x$lower, x$upper)),
    pch = 19,
    xlab = "Log Lambda1",
    ylab = "Mean Loss",
    main = "Mean Loss vs. Lambda"
  )
  arrows(
    log(x$lambda1),
    x$lower,
    log(x$lambda1),
    x$upper,
    length = 0.05,
    angle = 90,
    code = 3
  )
}

# Helper Functions for R

#' @title expit
#' @description expit function
#' @name expit
#' @param vector or scalar to use
#' @export
expit <- function(x) {
  return(exp(x) / (1 + exp(x)))
}

#' @title logit
#' @description logit function
#' @name logit
#' @param vector or scalar to use
#' @export
logit <- function(x) {
  return(log(x / (1 - x)))
}

#' @useDynLib gspam
#' @importFrom Rcpp sourceCpp
NULL
