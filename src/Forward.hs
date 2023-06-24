module Forward where

import Data.List (foldl)
import Types (Layer, Network, Neuron (..))

-- Calculate the output of a single neuron
calculateOutput :: Neuron -> [Float] -> Float
-- Apply the activation function to the sum of the weighted inputs and the bias
calculateOutput (Neuron weights bias (activation, _)) inputs =
  activation $ sum (zipWith (*) weights inputs) + bias

-- Calculate the outputs of a layer of neurons
calculateLayerOutputs :: Layer -> [Float] -> [Float]
 -- Apply 'calculateOutput' to each neuron in the layer
 -- The result is a list of outputs, one for each neuron
calculateLayerOutputs layer inputs = map (`calculateOutput` inputs) layer

-- Calculate the outputs of the entire network
calculateNetworkOutputs :: Network -> [Float] -> [Float]
 -- Fold over the layers of the network
 -- For each layer, calculate its outputs using the outputs of the previous layer as inputs
 -- The result is the outputs of the last layer of the network
calculateNetworkOutputs network inputs = foldl (flip calculateLayerOutputs) inputs network