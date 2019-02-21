# Define expit function
expit <- function(x){
  return(exp(x)/(1+exp(x)))
}


# GENERATE DATA
set.seed(1408)
# Set Params
# Number of obs
n <- 100
# Number of numeric features
p <- 100
# Number of numeric signal features
s <- 5
# Ratio of shape tuning param to sparsity tuning param
cats <- c("yes", "no","maybe")

cat_data <- sample(cats, n, replace =T)
cats <-rownames(table(cat_data))
numdata<-match(cat_data,cats)-1
# X is random normal
X <- matrix(rnorm(n*p),ncol=p)
beta <- c(rep(1,s),rep(0,p-s))
sig <- sign(X)
true_signal <- sig %*% beta+(numdata== 1)-(numdata == 2)
y_binary <- rbinom(n,size=1,prob=expit(true_signal))
y_continuous <- rnorm(n,mean = true_signal, sd = 1 )

data <- as.data.frame(X)
example_data <- cbind(data,cat_data,true_signal,y_binary,y_continuous)

usethis::use_data(example_data)
