module Training where

import Backward (calculateNetworkErrorDeltas)
import Control.Monad.Random ( Rand, RandomGen, RandT )
import Forward (calculateNetworkOutputs)
import Loss (crossEntropyLoss)
import Types (Inputs, LearningRate, Network, Target, BatchSize)
import Utility (oneHotEncode, shuffle, updateNetworkBatch, mean, shuffle')
import System.Random ()
import Data.List ( zipWith3 )
import Data.Maybe ( fromJust )
import Data.List.Split (chunksOf)

-- | Train the network over multiple epochs.
train' :: (RandomGen g, MonadIO m) => Network -> BatchSize -> [(Inputs, Target)] -> [(Inputs, Target)] -> LearningRate -> Int -> RandT g m Network
train' network batchSize trainingData validationData learningRate epochs =
  foldl' (trainAndValidateEpoch batchSize trainingData validationData learningRate) (return network) [1 .. epochs]

trainAndValidateEpoch :: (RandomGen g, MonadIO m) => BatchSize -> [(Inputs, Target)] -> [(Inputs, Target)] -> LearningRate -> RandT g m Network -> Int -> RandT g m Network
trainAndValidateEpoch batchSize trainingData validationData learningRate networkM _ = do
  network <- networkM
  shuffledData <- shuffle' trainingData
  let network' = foldl' (trainMiniBatch learningRate) network (miniBatches batchSize shuffledData)
  let validationLoss = computeLoss network' validationData
  liftIO $ putStrLn $ "Epoch complete. Validation loss: " ++ show validationLoss
  return network'

computeLoss :: Network -> [(Inputs, Target)] -> Float
computeLoss network data1 = 
  let (inputs, targets) = unzip data1
      maybeOutputs = mapM (calculateNetworkOutputs network) inputs
      losses = case maybeOutputs of
                 Just outputs ->
                   let targets' = map oneHotEncode targets
                       flattenedOutputs = map (fromJust . viaNonEmpty last) outputs
                   in zipWith crossEntropyLoss targets' flattenedOutputs
                 Nothing -> []
  in mean losses


-- | Train the network for one epoch.
trainEpoch :: RandomGen g => [(Inputs, Target)] -> LearningRate -> Rand g Network -> Int -> Rand g Network
trainEpoch trainingData learningRate networkM _ = do
  network <- networkM
  shuffledData <- shuffle trainingData
  let miniBatchedData = miniBatches 10 shuffledData -- The batch size is set to 10, but you can adjust this
  return $ concatMap (trainMiniBatch learningRate network) miniBatchedData

-- | Split the training data into mini-batches of a given size.
miniBatches :: Int -> [(Inputs, Target)] -> [[(Inputs, Target)]]
miniBatches = chunksOf

-- | Train the network on a mini-batch of training examples.
trainMiniBatch :: LearningRate -> Network -> [(Inputs, Target)] -> Network
trainMiniBatch learningRate network miniBatch =
  let (inputs, targets) = Prelude.unzip miniBatch
      maybeOutputs = mapM (calculateNetworkOutputs network) inputs
   in case maybeOutputs of
        Just outputs ->
          let targets' = map oneHotEncode targets
              flattenedOutputs = map (fromJust . viaNonEmpty last) outputs
              _losses = Prelude.zipWith crossEntropyLoss targets' flattenedOutputs
              maybeDeltas = sequence $ Data.List.zipWith3 calculateNetworkErrorDeltas (repeat network) targets' outputs -- use `outputs` here
           in case maybeDeltas of
                Just deltas ->
                  let inputsAndDeltas = Prelude.zip inputs deltas
                   in updateNetworkBatch learningRate network inputsAndDeltas
                Nothing -> network
        Nothing -> network