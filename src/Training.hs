module Training where

import Backward (calculateNetworkErrorDeltas)
import Control.Monad.Random ( runRand, Rand, RandomGen )
import Forward (calculateNetworkOutputs)
import Loss (crossEntropyLoss)
import Types (Inputs, LearningRate, Network, Target)
import Utility (oneHotEncode, shuffle, updateNetwork, updateNetworkBatch)
import System.Random ()
import Data.List ( zipWith3 )

-- | Train the network over multiple epochs.
train :: RandomGen g => Network -> [(Inputs, Target)] -> LearningRate -> Int -> g -> (Network, g)
train network trainingData learningRate epochs = runRand (train' network trainingData learningRate epochs)

train' :: RandomGen g => Network -> [(Inputs, Target)] -> LearningRate -> Int -> Rand g Network
train' network trainingData learningRate epochs =
  foldl' (trainEpoch trainingData learningRate) (return network) [1 .. epochs]

-- | Train the network for one epoch.
trainEpoch :: RandomGen g => [(Inputs, Target)] -> LearningRate -> Rand g Network -> Int -> Rand g Network
trainEpoch trainingData learningRate networkM _ = do
  network <- networkM
  shuffledData <- shuffle trainingData
  return $ foldl' (trainMiniBatch learningRate) network (miniBatches shuffledData)

-- | Split the training data into mini-batches.
miniBatches :: [(Inputs, Target)] -> [[(Inputs, Target)]]
miniBatches = undefined -- Implement this function

-- | Train the network on a mini-batch of training examples.
-- | Train the network on a mini-batch of training examples.
trainMiniBatch :: LearningRate -> Network -> [(Inputs, Target)] -> Network
trainMiniBatch learningRate network miniBatch =
  let (inputs, targets) = unzip miniBatch
      maybeOutputs = mapM (calculateNetworkOutputs network) inputs
   in case maybeOutputs of
        Just outputs ->
          let targets' = map oneHotEncode targets
              _losses = zipWith crossEntropyLoss targets' outputs
              maybeDeltas = sequence $ zipWith3 calculateNetworkErrorDeltas (repeat network) targets' outputs
           in case maybeDeltas of
                Just deltas -> foldl' (updateNetworkBatch learningRate) network deltas
                Nothing -> network
        Nothing -> network