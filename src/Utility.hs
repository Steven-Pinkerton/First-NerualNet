{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use one" #-}
module Utility where
import Types ( Layer, Neuron(weights, bias), ActivationFunction, LearningRate, Network, Inputs )
import System.Random ( RandomGen )
import Control.Monad.Random
    ( Rand )
import System.Random.Shuffle (shuffleM)
import Data.List ( zipWith3 )

getWeights :: Layer -> [Float]
getWeights = concat . map weights


-- The sigmoid activation function
sigmoid :: Float -> Float
sigmoid x = 1 / (1 + exp (- x))

-- The derivative of the sigmoid function
sigmoid' :: Float -> Float
sigmoid' x = let sx = sigmoid x in sx * (1 - sx)

-- Create an activation function
createActivationFunction :: String -> ActivationFunction
createActivationFunction "sigmoid" = (sigmoid, sigmoid')
createActivationFunction _ = error "Unknown activation function"

-- One-hot encode a digit
oneHotEncode :: Int -> [Float]
oneHotEncode n = [fromIntegral $ fromEnum (i == n) | i <- [0 .. 9]]

-- Shuffle a list
-- Shuffles a list, returning both the shuffled list and a new generator
shuffle :: RandomGen g => [a] -> Rand g [a]
shuffle = shuffleM

updateNetwork :: LearningRate -> [[[Float]]] -> [[Float]] -> Network -> Network
updateNetwork learningRate = zipWith3 (updateLayer learningRate)

updateLayer :: LearningRate -> [[Float]] -> [Float] -> Layer -> Layer
updateLayer learningRate = zipWith3 (updateNeuron learningRate)

updateNeuron :: LearningRate -> [Float] -> Float -> Neuron -> Neuron
updateNeuron learningRate neuronInputs delta neuron =
  neuron
    { weights = zipWith (\w x -> w - learningRate * delta * x) (weights neuron) neuronInputs
    , bias = bias neuron - learningRate * delta
    }


updateNetworkBatch :: LearningRate -> Network -> [(Inputs, [[Float]])] -> Network
updateNetworkBatch learningRate network inputsAndDeltas =
  let (allInputs, allDeltas) = unzip inputsAndDeltas
   in updateNetwork learningRate allDeltas allInputs network