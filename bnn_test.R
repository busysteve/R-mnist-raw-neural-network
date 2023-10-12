


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


# convert training labels from single digit number to ten digits of 0's and a 1
R <- lapply( labels, out10 )
R <- unlist(R)
R <- matrix( R, ncol=10, byrow = TRUE)

# convert testing labels from single digit number to ten digits of 0's and a 1
Rt <- lapply( labels, out10 )
Rt <- unlist(R)
Rt <- matrix( R, ncol=10, byrow = TRUE)



X <- I
y <- R

Sys.setenv(OMP_NUM_THREADS=7) 


nn <- bnn_create( name="mnist", inputs=784, hiddens=c( 600, 250, 100), outputs=10, hidden_acts=c("tanh", "tanh", "tanh"), output_act="tanh" )
nn <- bnn_trainer( nn, dataset=X, labels=y, epochs=1000, start_rate=.001, min_rate=.000001, batch_size=128, dropout=.05, dropout_mod=15, filename="network2.nn" )

bnn_store( nn, filename="my_nn.nn" )

nn2 <- bnn_load( filename="my_nn.nn" )





input <- X[345,]
result <- bnn_predict( nn2, input )

matrix( unlist(lapply( input, function(x) { if(x < .3) {' '} else{'#'}  } )) , ncol=28, byrow=TRUE)

bnn_choice( bnn_softmax(result) )




setwd("./testrun2")
# save weights and biases to files for testing
for( l in 1:( length(layer_size)-1 ) )
{
  write.table( w_layer[[l]], file=paste("./layer-weights-", l, sep="") )
  write.table( b_layer[[l]], file=paste("./bias-weights-", l, sep="") )
}


correct <- 0
for( it in 1:10000 )
{
  item <- it
  
  #X <- I[item,]
  #y <- labels[item]
  
  X <- It[item,]
  y <- labels.test[item]
  
  # display actual digit
  # matrix( unlist(lapply( dataset[item,], function(x) { if(x < .3) {' '} else{'#'}  } )) , ncol=28, byrow=TRUE)  
  
  # Forward propagation
  num_samples <- 1
  
  # Forward propagation
  result <- bnn_predict( nn, X )
  
  r <- bnn_choice( bnn_softmax( result ) )
  if( r == y )
  {
    correct = correct + 1
  }
}
correct






# test single sample from testing set

# I <- dataset
# R <- labels
I <- dataset.test
R <- labels.test



item = 3315
X = It[item,]
R[item]
matrix( unlist(lapply( X, function(x) { if(x < .3) {' '} else{'#'}  } )) , ncol=28, byrow=TRUE)  

# Forward propagation
num_samples <- 1

next_in[[1]] <- X %*% w_layer[[1]] + matrix(rep(b_layer[[1]], num_samples), nrow = num_samples, byrow = TRUE) 
next_out[[1]] <- do.call( acts[[1]], list(next_in[[1]]) ) 
for( l in 2:(length(layer_size) -1 ) )
{
  next_in[[l]] <- next_out[[l-1]] %*% w_layer[[l]] + matrix(rep(b_layer[[l]], num_samples), nrow = num_samples, byrow = TRUE) 
  next_out[[l]] <- do.call( acts[[l]], list(next_in[[l]]) ) 
}

output_output <- next_out[[l]]
softmax( output_output )
choice( softmax( output_output ) )





