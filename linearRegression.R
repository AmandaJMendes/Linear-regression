library(iterators)

X <- seq(-3, 3, by = 0.1)
f <- 2*X - 1
Y <- f + 0.5*rnorm(length(X), mean = 0, sd = 1)

batch_size <- 16
epochs <- 100
lr <- 0.1
dataset <- data.frame(X = X, Y = Y)
n_batches <- ceiling(nrow(dataset)/batch_size)
batches <- split(dataset, sample(1:n_batches, nrow(dataset), replace=T))

w <- rnorm(1, mean = 0, sd = 1)
bias <- rnorm(1, mean = 0, sd = 1)


for (epoch in 0:epochs){
  for (batch in batches){
    x <- batch$X
    y <- batch$Y
    
    yhat <- w*x + bias
    loss <- sum((yhat - y)**2)/length(x)
    grad_w <- (2/length(x)) * sum((y - w*x - bias)*-x)
    grad_bias <- (2/length(x)) * sum((y - w*x - bias)*-1)
    w <- w - lr*grad_w
    bias <- bias - lr*grad_bias
  }
}

print("Original function: y = 2x - 1")
print(paste("Learned function: y = ", round(w, 4), "x + ", round(bias,4), sep= ""))
