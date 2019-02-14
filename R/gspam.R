#' # Categorical Variables Prox Solver
#' #' @title Categorical Variable Prox Solver
#' #' @description Build and fit a sparsity penalized categorical variable
#' #' @name gspam_cat
#' #' @param data covariate list of strings (or matrix of strings)
#' #' @param y response variable numeric values
#' #' @param lambda sparsity penalty
#' #' @export
#' gspam_cat<-function(data=data,y=y,lambda=lambda){
#'   if(is.matrix(data)){
#'     numdata<-matrix(nrow=nrow(data),ncol=ncol(data))
#'     countdata<-c()
#'     for(i in 1:ncol(data)){
#'       cat_obj<-gspam_cat_build(data[,i])
#'       numdata[,i]=cat_obj$numdata
#'       countdata=coundata+cat_obj$cats$freq
#'     }
#'
#'     fitted<-gspam_cat_solver(numdata,y,countdata,"cat",lambda)
#'   }
#'   else if(is.vector(data)){
#'     cat_obj<-gspam_cat_build(data)
#'     fitted<-gspam_cat_solver(matrix(cat_obj$numdata,ncol=1),y,matrix(cat_obj$cats$Freq,ncol=1),"cat",lambda)
#'   }
#'   else{stop("Bad data type for cat (use matrix or vector)")}
#'   return(fitted)
#' }

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
  newlist<-list("freq"=cats$Freq,"numdata"=numdata)
  return( newlist)
}

# Categorical Variable Builder
#' @title Predict Function
#' @description Build a categorical variable
#' @name predict
#' @param data object returned by gspam for a single pair of lambda values
#' @export
predict<-function(data=data,new_point){
  fitted_point<-interpolate(data$data,data$fitted,new_point)
  return(fitted_point)
}


#' # Generalized Sparse Additive Model Solver
#' #' @title GSpam solver
#' #' @description Wrapper for Vspam solvers, can solve on matrix or data frame for single or path of lambdas.
#' #' @name GSpam
#' #' @param data covariate matrix/data frame
#' #' @param y response vector
#' #' @param prox_type vector of prox types to use
#' #' @param lambda1 lambda value(s)
#' #' @param lambda2 lambda value(s)
#' #'
#' #' @export
#' gspam<-function(data,y,prox_type,lambda1,lambda2){
#' data_c<-matrix(nrow=nrow(data),ncol=ncol(data))
#'     for(i in 1:ncol(data)){
#'       if(prox_type[i]!="cat"){
#'       data_c[,i]=data[,i]
#'     }
#'     else{
#'       temp<-gspam_cat_build(data[i])
#'       data_c[,i]=temp$numdata
#'     }
#'     }
#'   if(length(lambda1)!=length(lambda2)){
#'     error("lambda vectors must have same length")
#'   }
#'   if(length(lambda1)>1){
#'     fit<-gspam_c_vec(data_c,y,prox_type,lambda1,lambda2)
#'   }
#'  else{
#'    fit<-gspam_c(data_c,y,prox_type,lambda1,lambda2)
#'  }
#'
#'   return(fit)
#' }


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
