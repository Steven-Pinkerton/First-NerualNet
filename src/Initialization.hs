module Initialization where

import System.Random (RandomGen, randomR)
import Types (Network, Neuron (..))
import Utility (createActivationFunction)

{- | Initialize a neuron with Xavier initialization for weights and zero initialization for bias.
 The Xavier initialization helps to ensure that the weights are not too small or too large at the start of the training.
 This makes the training more efficient.
-}
initializeNeuron :: RandomGen g => g -> Int -> (Neuron, g)
initializeNeuron g nInputs =
  let (weights, g') = initializeWeights g nInputs -- generate weights
      bias = 0 -- initialize bias to zero
      activationFunction = createActivationFunction "sigmoid" -- create the activation function
   in (Neuron weights bias activationFunction, g')

{- | Initialize weights with Xavier initialization.
 The Xavier initialization generates random weights from a uniform distribution with limits ±sqrt(1/n),
 where n is the number of inputs to the neuron.
-}
initializeWeights :: RandomGen g => g -> Int -> ([Float], g)
initializeWeights g nInputs =
  let limit = sqrt (1 / fromIntegral nInputs) -- calculate the limit for the Xavier initialization
      initWeight (ws, g') = let (w, g'') = randomR (- limit, limit) g' in (w : ws, g'')
   in iterateN nInputs initWeight ([], g)

-- Repeat a function n times
iterateN :: Int -> (a -> a) -> a -> a
iterateN n f = foldr (.) id (replicate n f)

{- | Initialize a layer of neurons.
 A layer is simply a list of neurons, and the number of neurons in the layer is defined by the user.
-}
initializeLayer :: RandomGen g => g -> Int -> Int -> ([Neuron], g)
initializeLayer g nNeurons nInputs =
  let initNeuron (ns, g') = let (n, g'') = initializeNeuron g' nInputs in (n : ns, g'')
   in iterateN nNeurons initNeuron ([], g)

-- | Initialize a network of layers.
-- A network is a list of layers, and the number of layers in the network is defined by the user.
-- This function takes a list of integers, where each integer specifies the number of neurons in a layer.
initializeNetwork :: RandomGen g => g -> [Int] -> (Network, g)
initializeNetwork g layerSizes =
  let sizes = zip (0:layerSizes) layerSizes  -- pair each layer size with the next one
      initLayer (ns, g') (nInputs, nNeurons) = let (layer, g'') = initializeLayer g' nNeurons nInputs in (layer:ns, g'')
  in foldl' initLayer ([], g) sizes