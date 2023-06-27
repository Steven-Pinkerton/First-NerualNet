module Backward where

import Types (Layer, Network, Neuron (..))
import Data.List (zipWith3)
import Utility ( getWeights )

-- Compute the error of a single neuron in the output layer
calculateOutputError :: Float -> Float -> Float
calculateOutputError target output = output - target

-- Compute the delta of a single neuron in the output layer
calculateOutputDelta :: Float -> Float -> Float
calculateOutputDelta error' activationDerivative = error' * activationDerivative

-- Compute the error of a single neuron in a hidden layer
calculateHiddenError :: Float -> Float -> Float -> Float
calculateHiddenError nextWeight nextDelta activationDerivative =
  nextWeight * nextDelta * activationDerivative

-- Compute the delta of a single neuron in a hidden layer
calculateHiddenDelta :: Float -> Float -> Float
calculateHiddenDelta error' activationDerivative = error' * activationDerivative

-- Update a single weight
updateWeight :: Float -> Float -> Float -> Float -> Float
updateWeight weight learningRate delta neuronOutput =
  weight - learningRate * delta * neuronOutput

-- Update a single bias
updateBias :: Float -> Float -> Float -> Float
updateBias bias' learningRate delta = bias' - learningRate * delta

-- Compute the errors and deltas for a layer
calculateLayerErrorDeltas :: Layer -> [Float] -> [Float] -> [Float] -> [Float]
calculateLayerErrorDeltas layer nextDeltas nextWeights outputs =
  let activationDerivatives = zipWith snd (map (snd . activationFunctions . activationType) layer) outputs
   in zipWith3 calculateHiddenError nextWeights nextDeltas activationDerivatives

-- Compute the errors and deltas for the network
calculateNetworkErrorDeltas :: Network -> [Float] -> [[Float]] -> Maybe [[Float]]
calculateNetworkErrorDeltas network targets outputs = do
  lastOutputs <- viaNonEmpty last outputs
  lastNetwork <- viaNonEmpty last network
  let outputErrors = zipWith calculateOutputError targets lastOutputs
  let outputDeltas = zipWith calculateOutputDelta outputErrors (zipWith (\n output -> let (_, df) = activationFunctions $ activationType n in df output) lastNetwork lastOutputs)
  hiddenLayers <- viaNonEmpty init network
  hiddenOutputs <- viaNonEmpty init outputs
  let networkWeights = map getWeights hiddenLayers -- new line here to get the weights
  Just $
    reverse $
      foldr
        ( \(layer, layerOutputs, nextWeights) acc -> case viaNonEmpty head acc of
            Just accHead -> calculateLayerErrorDeltas layer layerOutputs accHead nextWeights : acc
            Nothing -> acc
        )
        [outputDeltas]
        (zip3 hiddenLayers hiddenOutputs networkWeights) -- change zip to zip3