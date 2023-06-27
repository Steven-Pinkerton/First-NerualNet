module Main where

import Training ( train' ) -- import necessary modules
import Initialization ( initializeNetwork )

main :: IO ()
main = do
  -- 1. Load data
  trainingData <- loadTrainingData
  validationData <- loadValidationData

  -- 2. Preprocess data
  let processedTrainingData = preprocess trainingData
  let processedValidationData = preprocess validationData

  -- 3. Initialize a random neural network
  -- Here, you'll have to define the structure of your network, i.e., the number of layers and neurons in each layer
  -- This will depend on the architecture you want to use. For example, for a simple, fully connected network,
  -- you might have a single hidden layer with a size somewhere between the input layer size (784 for MNIST) and
  -- the output layer size (10 for MNIST, since we have 10 digits).
  let network = initializeNetwork

  -- 4. Train the network
  -- Choose the learning rate, batch size and number of epochs that work best for you
  let learningRate = 0.01
  let batchSize = 10
  let epochs = 30
  trainedNetwork <- train' network batchSize processedTrainingData processedValidationData learningRate epochs

  -- 5. Evaluate the network
  let testPerformance = evaluate trainedNetwork validationData
  print testPerformance

  -- 6. Save the model
  saveModel trainedNetwork "model_path"