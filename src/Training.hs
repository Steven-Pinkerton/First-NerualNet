module Training where

import Backward
    ( calculateOutputError, calculateOutputDelta, updateBias )
import Forward ( calculateNetworkOutputs )
import Loss ()
import Types ( Network, Neuron(bias), Inputs, Target, Outputs )

-- Define the forward pass
forwardPass :: Network -> Inputs -> Outputs
forwardPass = calculateNetworkOutputs

-- Define the backward pass
backwardPass :: Network -> Outputs -> Target -> Network
backwardPass network outputs target =
  let learningRate = 0.01 -- you might want to define this elsewhere
      outputErrors = zipWith calculateOutputError target outputs
      outputDeltas = map calculateOutputDelta outputErrors
      newNetwork = zipWith (\neuron delta -> neuron {bias = updateBias (bias neuron) learningRate delta}) network outputDeltas
   in newNetwork

-- Define the main training loop
trainNetwork :: Network -> [(Inputs, Target)] -> Float -> Int -> Network
trainNetwork network trainingData learningRate epochs =
  let updatedNetwork = foldl' trainOnce network trainingData
   in if epochs > 0
        then trainNetwork updatedNetwork trainingData learningRate (epochs - 1)
        else updatedNetwork
  where
    -- Function to train the network on a single (input, target) pair
    trainOnce :: Network -> (Inputs, Target) -> Network
    trainOnce networkk (inputs, target) =
      let outputs = forwardPass networkk inputs
          network' = backwardPass networkk outputs target
       in network'