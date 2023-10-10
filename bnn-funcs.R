

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

bnn_outs <- function(x, cnt) {
  v <- t(((rep(0.0, cnt))))
    
  v[x+1]=1

  v
}

# Define the sigmoid activation function and its derivative
bnn_sigmoid <- function(x) {
  return(1 / (1 + exp(-(ifelse( x > .000001, .000001, x )))))
}

bnn_sigmoid_derivative <- function(x) {
  return(x * (1 - x))
}

# Define the Hyperbolic Tangent (tanh) activation function and its derivative
bnn_tanh <- function(x) {
  a <- (exp(x) + exp(-x))
  b <- (exp(x) - exp(-x)) / ifelse( a == 0, .000001, a )
  ifelse( is.nan(b), 1, b )
}

# Derivative of tanh activation function
bnn_tanh_derivative <- function(x) {
  1 - tanh(x)^2
}

# LRU (Leaky Rectified Linear Unit) activation function and its derivative
bnn_lrlu <- function(x, alpha = 0.01) {
  ifelse(x > 0, x, alpha * x)
}

bnn_lrlu_derivative <- function(x, alpha = 0.01) {
  ifelse(x > 0, 1, alpha)
}

# ReLU (Rectified Linear Unit) activation function and its derivative
bnn_relu <- function(x) {
  ifelse(x > 0, x, 0)
}

bnn_relu_derivative <- function(x) {
  ifelse(x > 0, 1, 0)
}

# ReLU (Rectified Linear Unit) activation function and its derivative
bnn_linear <- function(x) {
  x
}

bnn_linear_derivative <- function(x) {
  1
}

# softmax for multi-output networks
bnn_softmax <- function(x) {
  e_x <- exp(x - max(x))
  return(e_x / sum(e_x))
}

# Define the cross-entropy cost function
bnn_cross_entropy <- function(y_true, y_pred) {
  # Avoid log(0) by adding a small epsilon value
  epsilon <- 1e-15
  
  # Clip predicted values to avoid log(0) or log(1) issues
  y_pred <- pmax(epsilon, pmin(1 - epsilon, y_pred))
  
  # Calculate the cross-entropy loss
  loss <- -sum(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
  
  return(loss)
}

# weight matrix masking function
bnn_dropout <- function(m,c,b=TRUE){
  if( b == FALSE )
  {
    return(m)
  }
  d=dim(m)
  w=d[1]*d[2]
  x=as.integer(w*c)
  y=w-x
  return( m * matrix( sample(append(rep(0,x),rep(1,y))), nrow=d[1], ncol=d[2] ) )
}



# set activation functions for our layers
bnn_set_activations <- function(n)
{
  
  n$acts <- list()
  for( a in 1:length(n$hidden_act) )
  {
    n$acts[[a]] <- ( switch( n$hidden_act[a],
                                "tanh" = bnn_tanh, "sigmoid" = bnn_sigmoid, "relu" = bnn_relu, "lru" = bnn_lrlu, "linear" = bnn_linear ) )
  }
  n$acts[[a+1]] <- ( switch( n$output_act,
                              "tanh" = bnn_tanh, "sigmoid" = bnn_sigmoid, "relu" = bnn_relu, "lru" = bnn_lrlu, "linear" = bnn_linear  ) )
  
  n$derivs <- list()
  for( a in 1:length(n$hidden_act) )
  {
    n$derivs[[a]] <- ( switch( n$hidden_act[a],
                                "tanh" = bnn_tanh_derivative, "sigmoid" = bnn_sigmoid_derivative, "relu" = bnn_relu_derivative, "lru" = bnn_lrlu_derivative, "linear" = bnn_linear_derivative  ) )
  }
  n$derivs[[a+1]] <- ( switch( n$output_act,
                                "tanh" = bnn_tanh_derivative, "sigmoid" = bnn_sigmoid_derivative, "relu" = bnn_relu_derivative, "lru" = bnn_lrlu_derivative, "linear" = bnn_linear_derivative ) )

  n
}


bnn_set_weights <- function(n)
{
  n$layer_size <- append( append( n$input_size, n$hidden_size ), n$output_size )
  n$w_layer <- list()
  n$b_layer <- list()
  
  set.seed(proc.time()[3])  # For reproducibility
  
  for( l in 1:(length(n$layer_size)-1) )
  {
    n$w_layer[[l]] <- matrix(runif(n$layer_size[l] * n$layer_size[l+1], min=-1, max=1), nrow = n$layer_size[l], ncol = n$layer_size[l+1]) 
    n$b_layer[[l]] <- matrix(runif(n$layer_size[l+1], min=-1, max=1), nrow = 1, ncol = n$layer_size[l+1]) 
  }
  
  n
}



bnn_create <- function( name="mnist", inputs=784, hiddens=c(200, 100), outputs=10, hidden_acts=c("tanh", "tanh"), output_act="tanh")
{
  nn <- list()
  nn$name <- name
  nn$input_size <- inputs
  nn$hidden_size <- hiddens
  nn$hidden_acts <- hidden_acts
  nn$output_size <- outputs
  nn$output_act <- output_act
  nn <- bnn_set_activations(nn)
  nn <- bnn_set_weights(nn)

  nn  
}


bnn_predict <- function( nn, input )
{
  if( class(input) != "matrix")
    input <- matrix(input,ncol=length(input), byrow = TRUE)

  # Forward propagation
  num_batch_samples <- dim(input)[1]
  next_in <- list()
  next_out <- list()
  next_in[[1]] <- input %*% nn$w_layer[[1]] + matrix(rep(nn$b_layer[[1]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
  next_out[[1]] <- do.call( nn$acts[[1]], list(next_in[[1]]) ) 
  for( l in 2:(length(nn$layer_size) -1 ) )
  {
    next_in[[l]] <- next_out[[l-1]] %*% nn$w_layer[[l]] + matrix(rep(nn$b_layer[[l]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
    next_out[[l]] <- do.call( nn$acts[[l]], list(next_in[[l]]) ) 
  }
  
  next_out[[l]]
}



bnn_trainer <- function( nn, dataset, labelset, epochs=1000, start_rate=.005, min_rate=.000001, batch_size=128, dropout=0, dropout_mod=0 )
{
  num_samples <- dim(dataset)[1]
  
  tm <- proc.time()
  last_loss = 0
  
  
  batches1 = seq(from=1, to= num_samples, by=batch_size)
  batches2 = batches1 + batch_size-1
  batches2[ length(batches2) ] = num_samples
  batches = matrix( c(batches1,batches2), ncol=2, byrow=FALSE)
  
  learning_rate <- start_rate
  min_learning_rate <- min_rate
  
  min_loss = 3
  neg_rate_diff_tot = 0
  
  next_in = list()
  next_out = list()
  
  for (epoch in 1:epochs) {
  
    samples <- sample( 1:num_samples )
    
    losses = c(min_loss)
  
    
    for ( i in 1:dim(batches)[1] ) {
      
      
      batch = batches[i,]
      XX <- dataset[ samples[ batch[1]:batch[2] ],]
      yy <- labelset[ samples[ batch[1]:batch[2] ],]
      
      num_batch_samples = batch[2] - batch[1] + 1
      
      if( i >= 2 )
      {
        h=10
      }
      
      if( is.na( nn$w_layer[[1]][1,1] ) )
      {
        h = 3
      }
      
      #length( samples[ batch[1]:batch[2] ] )
      
      # Forward propagation
      next_in[[1]] <- XX %*% nn$w_layer[[1]] + matrix(rep(nn$b_layer[[1]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
      next_out[[1]] <- do.call( nn$acts[[1]], list(next_in[[1]]) ) 
      for( l in 2:(length(nn$layer_size) -1 ) )
      {
        next_in[[l]] <- next_out[[l-1]] %*% nn$w_layer[[l]] + matrix(rep(nn$b_layer[[l]], num_batch_samples), nrow = num_batch_samples, byrow = TRUE) 
        next_out[[l]] <- do.call( nn$acts[[l]], list(next_in[[l]]) ) 
      }
  
      # Compute the loss
      #loss <- ((sum((y - output_output)^2))) / (2*dim(output_output)[1])
      loss <- (sum(abs((yy - next_out[[l]])^2))) / (2*dim(next_out[[l]])[1])
      losses <- append(losses, loss)
      #loss <- cross_entropy( y, output_output ) / dim(output_output)[1]
      
      if( is.na(loss) )
      {
        h = 11
      }
      
      
      # Backpropagation
  
      l <- length(next_out)
      l_error <- list()
      l_delta <- list()
      l_error[[l]] <- -(yy - next_out[[l]])
      l_delta[[l]] <- l_error[[l]] * do.call(  nn$derivs[[l]], list(next_out[[l]]) )
      for( l in (l-1):1 )
      {
        l_error[[l]] <- l_delta[[l+1]] %*% t(nn$w_layer[[l+1]])
        l_delta[[l]] <- l_error[[l]] * do.call(  nn$derivs[[l]], list(next_out[[l]]) )
      }
      
      # Update weights and biases
      l <- length(nn$w_layer)
      for( l in l:2 )
      {
        nn$w_layer[[l]] <- nn$w_layer[[l]] - t(next_out[[l-1]]) %*% bnn_dropout( l_delta[[l]], dropout, i %% dropout_mod == 0) * learning_rate # %*% bnn_dropout(w_layer[[l]], dropout )
        nn$b_layer[[l]] <- nn$b_layer[[l]] - colSums(l_delta[[l]]) * learning_rate
      }
      
      nn$w_layer[[1]] <- nn$w_layer[[1]] - t(XX) %*% bnn_dropout( l_delta[[1]], dropout, i %% dropout_mod == 0 ) * learning_rate # %*% bnn_dropout(w_layer[[1]], dropout )
      nn$b_layer[[1]] <- nn$b_layer[[1]] - colSums(l_delta[[1]]) * learning_rate
      
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
  
  nn
}

# take a softmax output and determine the answer
bnn_choice <- function(x) {
  peak <- max(x)
  for( i in 1:dim(x)[2] ) {
    if( x[i] == peak ) {
      return(i)
    }
  }
}

# load weights and biases from files for testing
bnn_load_weights <- function(n)
{
  n$weights=list()
  n$biases=list()
  n$layer_cnt = 4
  for( l in 1:( layer_cnt-1 ) )
  {
    n$weights[[l]]<-as.matrix(read.table(file=paste("./layer-weights-", l, sep=""),header=T))
    n$biases[[l]]<-as.matrix(read.table(file=paste("./bias-weights-", l, sep=""),header=T))
  }
}
