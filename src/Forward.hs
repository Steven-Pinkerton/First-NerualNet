module Forward where

import Types (Layer, Network, Neuron (..))
import Utility ( activationFunctions )


-- Calculate the output of a single neuron
calculateOutput :: Neuron -> [Float] -> Float
calculateOutput (Neuron weights' bias' activationType') inputs =
  let (activation, _) = activationFunctions activationType'
   in activation $ sum (zipWith (*) weights' inputs) + bias'

-- Calculate the outputs of a layer of neurons
calculateLayerOutputs :: Layer -> [Float] -> [Float]
calculateLayerOutputs layer inputs = map (`calculateOutput` inputs) layer


calculateNetworkOutputs :: Network -> [Float] -> Maybe [[Float]]
calculateNetworkOutputs network inputs = Just $ scanl (flip calculateLayerOutputs) inputs network