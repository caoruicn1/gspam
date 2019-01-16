# TESTS
set.seed(1408)
# FL prox, Quad loss
# Num trials
N <- 1
# Set Params
n <- 1000
p <- 1000
s <- 5
alpha <- 0.5

# List of times
time_list <- c()
for(i in 1:N){
X <- matrix(rnorm(n*p),ncol=p)
beta <- c(rep(1,s),rep(0,p-s))
sig <- sign(X)
y <- rnorm(n)+sig %*% beta
prox_type <- rep("fl",p)
start.time <- Sys.time()
results <- gspam_full(X,y,prox_type,"quad",alpha)
end.time <- Sys.time()
time_list[i]<- end.time-start.time
}
mean(time_list)
# Verify that fit gets closer to actual y as lambda decreases.
rss <- c()
# Get Number of Active Sets
num_active <- c()

for(i in 1:100){
  rss[i] <- sum((y-rowSums(results$fitted[[i]]))^2)
  num_active[i]<- sum(colSums((results$fitted[[i]])[,-1]^2)>.0001)
}

plot(results$lambda1,num_active)




lambdas <- get_lambdas(X,y,prox_type,"quad",alpha)
lambda1 <- lambdas$lambda1
lambda2 <- lambdas$lambda2
# Randomly shuffle the data
randomizer <- sample(nrow(X))
rand_X <- X[randomizer,]
rand_y <- y[randomizer]

# Create k equally size folds
folds <- cut(seq(1,nrow(X)),breaks=10,labels=FALSE)

# Perform k fold cross validation
  testIndexes <- which(folds==1,arr.ind=TRUE)
  testX <- rand_X[testIndexes, ]
  testy <- rand_y[testIndexes]
  
  trainX <- rand_X[-testIndexes, ]
  trainy <- rand_y[-testIndexes]
  
result <- gspam_c_vec(trainX,trainy,prox_type,"quad",lambda1,lambda2)
test_fits <- c()
for(i in 1:100){
  fits <- c()
  for(j in 1: 1000){
  fits[j] <- interpolate(trainX,result$fitted[[i]],testX[j,])
  }
  test_fits[[i]] <- fits
}
mse <- c()
for(i in 1:1000){
  mse[i] <- mean((testy-test_fits[[i]])^2)
}



start.time <- Sys.time()
results <- gspam.cv(X,y,prox_type,"quad")
end.time <- Sys.time()
