# Categorical Variable Builder
#' @title Categorical Variable Prox builder
#' @description Build a categorical variable
#' @name gspam_cat_build
#' @param data covariate vector
#' @export
gspam_cat_build<-function(data=data){
  cats<-as.data.frame(table(data))
  nums<-c(0:(nrow(cats)-1))
  numdata<-match(data,cats$data)-1
  newlist<-list("numdata"=numdata,"freq"=cats$Freq)
  return( newlist)
}

#' @title Predict Function
#' @description Build a categorical variable
#' @name predict
#' @param data object returned by gspam for a single pair of lambda values
#' @export
predict<-function(data=data,new_point){
  fitted_point<-interpolate(data$data,data$fitted,new_point)
  return(fitted_point)
}


# Generalized Sparse Additive Model Solver
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
gspam.cv <- function(data,y,prox_type,loss_type,alpha=0.5,k=10){
  # Get lambda vector to do cv over frow full data
  lambdas <- get_lambdas(data,y,prox_type,loss_type,alpha)
  lambda1 <- lambdas$lambda1
  lambda2 <- lambdas$lambda2
  # Randomly shuffle the data
  randomizer <- sample(nrow(data))
  rand_X <- data[randomizer,]
  rand_y <- y[randomizer]
  
  # Create k equally size folds
  folds <- cut(seq(1,nrow(data)),breaks=k,labels=FALSE)
  
  # Perform k fold cross validation
  mselist <- matrix(rep(0,k*100),ncol=k)
  for(i in 1:k){
    testIndexes <- which(folds==k,arr.ind=TRUE)
    testX <- rand_X[testIndexes, ]
    testy <- rand_y[testIndexes]
    
    trainX <- rand_X[-testIndexes, ]
    trainy <- rand_y[-testIndexes]
    result <- gspam_c_vec(trainX,trainy,prox_type,"quad",lambda1,lambda2)
    test_fits <- c()
    for(i in 1:100){
      fits <- c()
      for(j in 1: nrow(testX)){
        fits[j] <- interpolate(trainX,result$fitted[[i]],testX[j,])
      }
      test_fits[[i]] <- fits
    }
    mse <- c()
    for(i in 1:100){
      mse[i] <- mean((testy-test_fits[[i]])^2)
    }
   print(mse)
   mselist[,k] <- mse
  }
  best_lambda1 <- lambda1[which.min(rowMeans(mselist))]
  lower <- rowQuantiles(mselist,.05)
  upper <- rowQuantiles(mselist,.95)
  return(list("mses" = rowMeans(mselist),"lambda1"=lambda1,"best_lambda1" = best_lambda1,lowermse = lower, uppermse= upper, full = mselist))
}




#' @useDynLib gspam
#' @importFrom Rcpp sourceCpp
NULL
