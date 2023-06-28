import Training ( evaluate, train' )
import Control.Monad.Random ( evalRandT, newStdGen )
import Initialization ( initializeNetwork )
import DataLoader
    ( loadTrainingData, loadValidationData )
import ModelIO ( saveModel )

main :: IO ()
main = do
  -- 1. Load and preprocess data
  processedTrainingData <- loadTrainingData
  processedValidationData <- loadValidationData

  -- 2. Initialize a random neural network
  gen <- newStdGen -- you need to create a new random generator
  let structure = [784, 16, 16, 10] -- example structure for a fully connected network with two hidden layers for MNIST
  let (network, _) = initializeNetwork gen structure -- initialize the network with the random generator and the structure

  -- 3. Train the network
  -- Choose the learning rate, batch size and number of epochs that work best for you
  let learningRate = 0.01
  let batchSize = 10
  let epochs = 30
  trainedNetwork <- evalRandT (train' network batchSize processedTrainingData processedValidationData learningRate epochs) gen

  -- 4. Evaluate the network
  let testPerformance = evaluate trainedNetwork processedValidationData -- use processedValidationData here
  print testPerformance

  -- 5. Save the model
  let modelPath = "trained_model.bin" -- Choose the path and filename for your saved model
  saveModel modelPath trainedNetwork