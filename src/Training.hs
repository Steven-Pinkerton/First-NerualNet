module Training where

import Backward (calculateNetworkErrorDeltas)
import Control.Monad.Random ( Rand, RandomGen, RandT )
import Forward (calculateNetworkOutputs)
import Loss (crossEntropyLoss)
import Types (Inputs, LearningRate, Network, Target, BatchSize)
import Utility (shuffle, updateNetworkBatch, mean, shuffle', argmax)
import System.Random ()
import Data.List ( zipWith3 )
import Data.Maybe ( fromJust )
import Data.List.Split (chunksOf)

{- | Train the network over multiple epochs.
 This function repeatedly applies `trainAndValidateEpoch` on a network over several epochs.
-}
train' :: (RandomGen g, MonadIO m) => Network -> BatchSize -> [(Inputs, Target)] -> [(Inputs, Target)] -> LearningRate -> Int -> RandT g m Network
train' network batchSize trainingData validationData learningRate epochs =
  foldl' (trainAndValidateEpoch batchSize trainingData validationData learningRate) (return network) [1 .. epochs]

{- | Train the network for one epoch and validate its performance.
 This function first shuffles the training data, then applies `trainMiniBatch` on the network for each mini-batch,
 calculates the validation loss, and then prints it out before returning the updated network.
-}
trainAndValidateEpoch :: (RandomGen g, MonadIO m) => BatchSize -> [(Inputs, Target)] -> [(Inputs, Target)] -> LearningRate -> RandT g m Network -> Int -> RandT g m Network
trainAndValidateEpoch batchSize trainingData validationData learningRate networkM _ = do
  network <- networkM
  shuffledData <- shuffle' trainingData
  let network' = foldl' (trainMiniBatch learningRate) network (miniBatches batchSize shuffledData)
  let validationLoss = computeLoss network' validationData
  liftIO $ putStrLn $ "Epoch complete. Validation loss: " ++ show validationLoss
  return network'

{- | Calculate the average loss of a network on some data.
 This function first calculates the network's output for each input in the data, then calculates the cross-entropy loss
 for each output-target pair, and finally calculates the mean of all losses.
-}
computeLoss :: Network -> [(Inputs, Target)] -> Float
computeLoss network data1 =
  let (inputs, targets) = unzip data1
      maybeOutputs = mapM (calculateNetworkOutputs network) inputs
      losses = case maybeOutputs of
        Just outputs ->
          let flattenedOutputs = map (fromJust . viaNonEmpty last) outputs
           in zipWith crossEntropyLoss targets flattenedOutputs
        Nothing -> []
   in mean losses

{- | Train the network for one epoch.
 This function first shuffles the training data, then splits it into mini-batches,
 and then applies `trainMiniBatch` on the network for each mini-batch.
-}
trainEpoch :: RandomGen g => [(Inputs, Target)] -> LearningRate -> Rand g Network -> Int -> Rand g Network
trainEpoch trainingData learningRate networkM _ = do
  network <- networkM
  shuffledData <- shuffle trainingData
  let miniBatchedData = miniBatches 10 shuffledData -- The batch size is set to 10, but you can adjust this
  return $ concatMap (trainMiniBatch learningRate network) miniBatchedData

{- | Split the training data into mini-batches of a given size.
 This function simply splits a list into chunks of a specific size.
-}
miniBatches :: Int -> [(Inputs, Target)] -> [[(Inputs, Target)]]
miniBatches = chunksOf

{- | Train the network on a mini-batch of training examples.
 This function first calculates the network's output for each input in the mini-batch,
 then calculates the error delta for each output-target pair, and then updates the network using these deltas.
-}
trainMiniBatch :: LearningRate -> Network -> [(Inputs, Target)] -> Network
trainMiniBatch learningRate network miniBatch =
  let (inputs, targets) = unzip miniBatch
      maybeOutputs = mapM (calculateNetworkOutputs network) inputs
   in case maybeOutputs of
        Just outputs ->
          let maybeDeltas = sequence $ zipWith3 calculateNetworkErrorDeltas (repeat network) targets outputs
           in case maybeDeltas of
                Just deltas ->
                  let inputsAndDeltas = zip inputs deltas
                   in updateNetworkBatch learningRate network inputsAndDeltas
                Nothing -> network
        Nothing -> network

{- | Evaluate the accuracy of a network on some test data.
 This function first calculates the network's output for each input in the test data,
 then compares these outputs to the expected targets to calculate the accuracy.
-}
evaluate :: Network -> [(Inputs, Target)] -> Float
evaluate network testData =
  let (inputs, targets) = unzip testData
      maybeOutputs = mapM (calculateNetworkOutputs network) inputs
   in case maybeOutputs of
        Just outputs ->
          let predictedTargets = map (argmax . fromJust . viaNonEmpty last) outputs
              actualTargets = map argmax targets
              comparisons = zipWith (==) actualTargets predictedTargets
              correctPredictions = fromIntegral $ length $ filter id comparisons
              totalPredictions = fromIntegral $ length comparisons
           in (correctPredictions / totalPredictions) * 100
        Nothing -> 0