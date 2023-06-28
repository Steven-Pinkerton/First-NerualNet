import Training ( evaluate, train' )
import Control.Monad.Random ( evalRandT, newStdGen )
import Initialization ( initializeNetwork )
import DataLoader
    ( loadTrainingData, loadValidationData, preprocess )

main :: IO ()
main = do
  -- 1. Load data
  trainingData <- loadTrainingData "training_data_file_path"
  validationData <- loadValidationData "validation_data_file_path"

  -- 2. Preprocess data
  let processedTrainingData = preprocess trainingData
  let processedValidationData = preprocess validationData

  -- 3. Initialize a random neural network
  gen <- newStdGen -- you need to create a new random generator
  let structure = [784, 16, 16, 10] -- example structure for a fully connected network with two hidden layers for MNIST
  let (network, _) = initializeNetwork gen structure -- initialize the network with the random generator and the structure

  -- 4. Train the network
  -- Choose the learning rate, batch size and number of epochs that work best for you
  let learningRate = 0.01
  let batchSize = 10
  let epochs = 30
  trainedNetwork <- evalRandT (train' network batchSize processedTrainingData processedValidationData learningRate epochs) gen

  -- 5. Evaluate the network
  let testPerformance = evaluate trainedNetwork processedValidationData -- use processedValidationData here
  print testPerformance

-- 6. Save the model
-- saveModel function needs to be adjusted according to your needs.