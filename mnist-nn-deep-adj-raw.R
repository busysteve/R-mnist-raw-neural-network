
setwd("~/rdev/")

mnist.train.data = file("./mnist/train-images.idx3-ubyte", "rb")
readBin(mnist.train.data, integer(), n = 8, size = 2, endian = "little")
mnist.data = readBin(mnist.train.data, integer(), n = 28*28*60000, size = 1, signed = F, endian = "little")
dataset = matrix( data=mnist.data, nrow=60000, ncol=28*28, byrow=TRUE ) / 255 

dim(dataset)

mnist.train.labels = file("./mnist/train-labels.idx1-ubyte", "rb")
readBin(mnist.train.labels, integer(), n = 8, size = 1, signed = F, endian = "little")
mnist.labels = readBin(mnist.train.labels, integer(), n = 60000, size = 1, signed = F, endian = "little")
labels = matrix( data=mnist.labels, nrow=60000, ncol=1, byrow=TRUE )

dim(labels)


mnist.test.data = file("./mnist/t10k-images.idx3-ubyte", "rb")
readBin(mnist.test.data, integer(), n = 8, size = 2, endian = "little")
mnist.data.test = readBin(mnist.test.data, integer(), n = 28*28*10000, size = 1, signed = F, endian = "little")
dataset.test = matrix( data=mnist.data.test, nrow=10000, ncol=28*28, byrow=TRUE ) / 255 

dim(dataset.test)

mnist.test.labels = file("./mnist/t10k-labels.idx1-ubyte", "rb")
readBin(mnist.test.labels, integer(), n = 8, size = 1, signed = F, endian = "little")
mnist.labels.test = readBin(mnist.test.labels, integer(), n = 10000, size = 1, signed = F, endian = "little")
labels.test = matrix( data=mnist.labels.test, nrow=10000, ncol=1, byrow=TRUE )

dim(labels.test)

# assign training data to I
I <- dataset

# assign testing data to It
It <- dataset.test


# convert a label from single numbers to 10 output indicators
out10 <- function(x) {
  v <- t(((rep(0.0, 10))))
  if( x == 0)
  {
    v[10]=1
  }
  else
  {
    v[x]=1
  }
  v
}

# convert training labels from single digit number to ten digits of 0's and a 1
R <- lapply( labels, out10 )
R <- unlist(R)
R <- matrix( R, ncol=10, byrow = TRUE)

# convert testing labels from single digit number to ten digits of 0's and a 1
Rt <- lapply( labels, out10 )
Rt <- unlist(R)
Rt <- matrix( R, ncol=10, byrow = TRUE)


# Define the sigmoid activation function and its derivative
sigmoid <- function(x) {
  return(1 / (1 + exp(-(ifelse( x > .000001, .000001, x )))))
}

sigmoid_derivative <- function(x) {
  return(x * (1 - x))
}

# Define the Hyperbolic Tangent (tanh) activation function and its derivative
tanh <- function(x) {
  a <- (exp(x) + exp(-x))
  b <- (exp(x) - exp(-x)) / ifelse( a == 0, .000001, a )
  ifelse( is.nan(b), 1, b )
}

# Derivative of tanh activation function
tanh_derivative <- function(x) {
  1 - tanh(x)^2
}

# LRU (Leaky Rectified Linear Unit) activation function and its derivative
lrlu <- function(x, alpha = 0.01) {
  ifelse(x > 0, x, alpha * x)
}

lrlu_derivative <- function(x, alpha = 0.01) {
  ifelse(x > 0, 1, alpha)
}

# ReLU (Rectified Linear Unit) activation function and its derivative
relu <- function(x) {
  ifelse(x > 0, x, 0)
}

relu_derivative <- function(x) {
  ifelse(x > 0, 1, 0)
}

# ReLU (Rectified Linear Unit) activation function and its derivative
linear <- function(x) {
  x
}

linear_derivative <- function(x) {
  1
}

# softmax for multi-output networks
softmax <- function(x) {
  e_x <- exp(x - max(x))
  return(e_x / sum(e_x))
}

# Define the cross-entropy cost function
cross_entropy <- function(y_true, y_pred) {
  # Avoid log(0) by adding a small epsilon value
  epsilon <- 1e-15
  
  # Clip predicted values to avoid log(0) or log(1) issues
  y_pred <- pmax(epsilon, pmin(1 - epsilon, y_pred))
  
  # Calculate the cross-entropy loss
  loss <- -sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
  
  return(loss)
}

# weight matrix masking function
mask <- function(m,c){
  d=dim(m)
  w=d[1]*d[2]
  x=as.integer(w*c)
  y=w-x
  return( m * matrix( sample(append(rep(0,x),rep(1,y))), nrow=d[1], ncol=d[2] ) )
}



# Initialize weights and biases
input_size <- 784  # Replace with the actual input size
hidden_size <- c(16,16,16)
output_size <- 10  # Replace with the actual output size

hidden_act <- c("sigmoid", "tanh", "tanh" )
out_act <- "tanh"


# set activation functions for our layers
acts <- list()
for( a in 1:length(hidden_act) )
{
  acts[[a]] <- ( switch( hidden_act[a],
                              "tanh" = tanh, "sigmoid" = sigmoid, "relu" = relu, "lru" = lrlu, "linear" = linear ) )
}
acts[[a+1]] <- ( switch( out_act,
                            "tanh" = tanh, "sigmoid" = sigmoid, "relu" = relu, "lrlu" = lrlu, "linear" = linear  ) )

derivs <- list()
for( a in 1:length(hidden_act) )
{
  derivs[[a]] <- ( switch( hidden_act[a],
                              "tanh" = tanh_derivative, "sigmoid" = sigmoid_derivative, "relu" = relu_derivative, "lru" = lrlu_derivative, "linear" = linear_derivative  ) )
}
derivs[[a+1]] <- ( switch( out_act,
                              "tanh" = tanh_derivative, "sigmoid" = sigmoid_derivative, "relu" = relu_derivative, "lru" = lrlu_derivative, "linear" = linear_derivative ) )



layer_size <- append( append( input_size, hidden_size ), output_size )
w_layer <- list()
b_layer <- list()


set.seed(proc.time()[3])  # For reproducibility

for( l in 1:(length(layer_size)-1) )
{
  w_layer[[l]] <- matrix(runif(layer_size[l] * layer_size[l+1]), nrow = layer_size[l], ncol = layer_size[l+1]) 
  b_layer[[l]] <- matrix(runif(layer_size[l+1]), nrow = 1, ncol = layer_size[l+1]) 
}




# Define the learning rate
learning_rate <- 0.001
min_learning_rate <- 0.000001
# dead_weight_perc <- 0.1

# Define the number of training iterations
batch_size <- 128
epochs <- 1000

# Training data copy to X and y
X <- I
y <- R

dim(X)
dim(y)






num_samples <- dim(X)[1]

tm <- proc.time()
last_loss = 0


# Training loop

batches1 = seq(from=1, to= num_samples, by=batch_size)
batches2 = batches1 + batch_size-1
batches2[ length(batches2) ] = 60000
batches = matrix( c(batches1,batches2), ncol=2, byrow=FALSE)

min_loss = 3
neg_rate_diff_tot = 0

next_in = list()
next_out = list()

for (epoch in 1:epochs) {

  samples <- sample( 1:num_samples )
  
  losses = c(min_loss)

  
  for ( i in 1:dim(batches)[1] ) {
    
    
    batch = batches[i,]
    X <- I[ samples[ batch[1]:batch[2] ],]
    y <- R[ samples[ batch[1]:batch[2] ],]
    
    num_batch_samples = batch[2] - batch[1] + 1
     
    #length( samples[ batch[1]:batch[2] ] )
    
    # Forward propagation
    #next_in[[1]] = (X)

    #for( l in 1:(length(layer_size)-1) )
    
    dim(X %*% w_layer[[1]])
    dim(matrix(rep(b_layer[[1]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE))
    dim(b_layer[[1]])
    next_in[[1]] <- X %*% w_layer[[1]] + matrix(rep(b_layer[[1]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
    next_out[[1]] <- do.call( acts[[1]], list(next_in[[1]]) ) 
    #for( l in 2:(length(layer_size) -1 ) )
    for( l in 2:(length(layer_size) -1 ) )
    {
      #next_in[[l]] <- next_in[[l-1]] %*% w_layer[[l]] + matrix(rep(b_layer[[l]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
      #next_out[[l]] <- do.call( acts[[l]], list(next_in[[l]]) ) 
      
      next_in[[l]] <- next_out[[l-1]] %*% w_layer[[l]] + matrix(rep(b_layer[[l]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
      next_out[[l]] <- do.call( acts[[l]], list(next_in[[l]]) ) 
      
      # hidden3_input <- hidden2_output %*% w_hidden3 + matrix(rep(b_hidden3, num_batch_samples), nrow = num_batch_samples, byrow = TRUE)
      # hidden3_output <- act3(hidden3_input)
      
    }
  
    length( w_layer)
    
    dim(next_in[[l]])
    dim(w_layer[[l]])
    dim(matrix(rep(b_layer[[l]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE))
    dim(b_layer[[l]])
    # Compute the loss
    #loss <- ((sum((y - output_output)^2))) / (2*dim(output_output)[1])
    loss <- (sum(abs((y - next_out[[l]])^2))) / (2*dim(next_out[[l]])[1])
    losses <- append(losses, loss)
    #loss <- cross_entropy( y, output_output ) / dim(output_output)[1]
    
    # Backpropagation

    l <- length(next_out)
    l_error <- list()
    l_delta <- list()
    dim(next_out[[l]])
    length(next_out)
    dim(y)
    l_error[[l]] <- -(y - next_out[[l]])
    l_delta[[l]] <- l_error[[l]] * do.call(  derivs[[l]], list(next_out[[l]]) )
    for( l in (l-1):1 )
    {
      dim(l_delta[[l+1]])
      dim(t(w_layer[[l]]))
      l_error[[l]] <- l_delta[[l+1]] %*% t(w_layer[[l+1]])
      l_delta[[l]] <- l_error[[l]] * do.call(  derivs[[l]], list(next_out[[l]]) )
    }
    
    
    length(l_delta)
    length(w_layer)
    
    # Update weights and biases
    l <- length(w_layer)
    for( l in l:2 )
    {
      w_layer[[l]] <- w_layer[[l]] - t(next_out[[l-1]]) %*% l_delta[[l]] * learning_rate #* mask(w_output, dead_weight_perc)
      b_layer[[l]] <- b_layer[[l]] - colSums(l_delta[[l]]) * learning_rate
    }
    
    w_layer[[1]] <- w_layer[[1]] - t(X) %*% l_delta[[1]] * learning_rate #* mask(w_output, dead_weight_perc)
    b_layer[[1]] <- b_layer[[1]] - colSums(l_delta[[1]]) * learning_rate
    

    # Print the current loss
    #if (epoch %% 100 == 0) {
    #  tot <- proc.time() - tm
    #  cat("Epoch:", epoch, "Batch:", i, batch[1], batch[2], "Loss:", loss, "Learning Rate:", learning_rate, "RateDiff:", last_loss - loss, "Timing:", tot, "\n")
    #  tm <- proc.time()
    #}

  }
  
  loss <- mean(losses)
  rate_diff <- last_loss - loss
  
  #if (epoch %% 10 == 0) {
    tot <- proc.time()[3] - tm
    cat("Epoch:", epoch, "Loss:", loss, "Learning Rate:", learning_rate, "RateDiff:", rate_diff, "Timing:", tot, "\n")
    tm <- proc.time()[3]
  #}
    
    if( abs(rate_diff) > 1 )
      next
    
    if( rate_diff > 0 )
      neg_rate_diff_tot = 0

    if( neg_rate_diff_tot > 1 )
    {
      learning_rate <- learning_rate * .2
      learning_rate <- ifelse( learning_rate < min_learning_rate, min_learning_rate, learning_rate )
    }
    else if( rate_diff < 0 )
    {
      learning_rate <- learning_rate * 1.05
      learning_rate <- ifelse( learning_rate < min_learning_rate, min_learning_rate, learning_rate )

      if( rate_diff < 0 )
        neg_rate_diff_tot = neg_rate_diff_tot + 1
      
      if( neg_rate_diff_tot >= 4 )
        break
    }

    min_loss <- ifelse( min_loss > loss, loss, min_loss)

    if( loss > min_loss*1.02 )
    {
      cat( "Loss:", loss, ">", "Min_Loss", min_loss*1.005 )
        break
    }
    
    if( loss < .01 && rate_diff < .000001 )
      break
    
    if( learning_rate < .00001 )
      break
    
    last_loss = loss
    #min_loss = 1
}



#############################################
#
#      to train, run script from here up
#
#############################################



# take a softmax output and determine the answer
choice <- function(x) {
  peak <- max(x)
  for( i in 1:dim(x)[2] ) {
    if( x[i] == peak ) {
      return(i)
    }
  }
}



correct <- 0
for( it in 1:10000 )
{
  item <- it
  X <- It[item,]
  y <- labels.test[item]
  # display actual digit
  # matrix( unlist(lapply( dataset[item,], function(x) { if(x < .3) {' '} else{'#'}  } )) , ncol=28, byrow=TRUE)  
  
  # Forward propagation
  num_samples <- 1
  
  hidden1_input <- X %*% w_hidden1 + matrix(rep(b_hidden1, num_samples), nrow = num_samples, byrow = TRUE)
  hidden1_output <- act1(hidden1_input)
  
  hidden2_input <- hidden1_output %*% w_hidden2 + matrix(rep(b_hidden2, num_samples), nrow = num_samples, byrow = TRUE)
  hidden2_output <- act2(hidden2_input)
  
  hidden3_input <- hidden2_output %*% w_hidden3 + matrix(rep(b_hidden3, num_samples), nrow = num_samples, byrow = TRUE)
  hidden3_output <- act3(hidden3_input)
  
  output_input <- hidden3_output %*% w_output + matrix(rep(b_output, num_samples), nrow = num_samples, byrow = TRUE)
  output_output <- act4(output_input)
  
  #output_output
  #softmax( output_output )
  r <- choice( softmax( output_output ) )
  if( r == y )
  {
    correct = correct + 1
  }
}
correct



# I <- dataset
# R <- labels
I <- dataset.test
R <- labels.test



item = 13325
X = I[item,]
labels[item]
matrix( unlist(lapply( dataset[item,], function(x) { if(x < .3) {' '} else{'#'}  } )) , ncol=28, byrow=TRUE)  

# Forward propagation
num_samples <- 1

hidden1_input <- X %*% w_hidden1 + matrix(rep(b_hidden1, num_samples), nrow = num_samples, byrow = TRUE)
hidden1_output <- act1(hidden1_input)

hidden2_input <- hidden1_output %*% w_hidden2 + matrix(rep(b_hidden2, num_samples), nrow = num_samples, byrow = TRUE)
hidden2_output <- act2(hidden2_input)

output_input <- hidden2_output %*% w_output + matrix(rep(b_output, num_samples), nrow = num_samples, byrow = TRUE)
output_output <- act3(output_input)

output_output
softmax( output_output )
choice( softmax( output_output ) )



