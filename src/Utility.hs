{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use one" #-}
module Utility where
import Types ( Layer, Neuron(weights, bias), ActivationType (Sigmoid, ReLU), LearningRate, Network, Inputs )
import System.Random ( RandomGen )
import Control.Monad.Random
    ( Rand )
import System.Random.Shuffle (shuffleM)
import Data.List ( zipWith3, (!!), maximum, minimum )
import Control.Monad.Random.Class ( MonadRandom(getRandomRs) )

-- | Retrieves the weights from each neuron in a layer.
getWeights :: Layer -> [Float]
getWeights = concatMap weights

-- | The sigmoid activation function, used to introduce non-linearity to the network.
sigmoid :: Float -> Float
sigmoid x = 1 / (1 + exp (-x))

-- | The derivative of the sigmoid function, used during backpropagation to adjust the network's weights and biases.
sigmoid' :: Float -> Float
sigmoid' x = let sx = sigmoid x in sx * (1 - sx)

-- | Factory function to create an activation function given its name.
-- | Factory function to create an activation type given its name.
createActivationType :: String -> ActivationType
createActivationType "sigmoid" = Sigmoid
createActivationType "relu" = ReLU
createActivationType _ = error "Unknown activation function"

-- | One-hot encode a digit into a list of zeroes with a single one at the index of the digit.
oneHotEncode :: Int -> [Float]
oneHotEncode n = [fromIntegral $ fromEnum (i == n) | i <- [0 .. 9]]

-- | Shuffle a list using a random generator.
shuffle :: RandomGen g => [a] -> Rand g [a]
shuffle = shuffleM

-- | Update the network's weights and biases based on the calculated error delta.
updateNetwork :: LearningRate -> [[[Float]]] -> [[Float]] -> Network -> Network
updateNetwork learningRate = zipWith3 (updateLayer learningRate)

-- | Update the weights and biases of a single layer based on the error delta.
updateLayer :: LearningRate -> [[Float]] -> [Float] -> Layer -> Layer
updateLayer learningRate = zipWith3 (updateNeuron learningRate)

-- | Update the weights and biases of a single neuron based on the error delta.
updateNeuron :: LearningRate -> [Float] -> Float -> Neuron -> Neuron
updateNeuron learningRate neuronInputs delta neuron =
  neuron
    { weights = zipWith (\w x -> w - learningRate * delta * x) (weights neuron) neuronInputs
    , bias = bias neuron - learningRate * delta
    }

-- | Update the network weights and biases using multiple batches of inputs and their respective error deltas.
updateNetworkBatch :: LearningRate -> Network -> [(Inputs, [[Float]])] -> Network
updateNetworkBatch learningRate network inputsAndDeltas =
  let (allInputs, allDeltas) = unzip inputsAndDeltas
   in updateNetwork learningRate allDeltas allInputs network

-- | Calculate the mean of a list of Floats.
mean :: [Float] -> Float
mean xs = sum xs / fromIntegral (length xs)

-- | Shuffle a list in a random order, using a MonadRandom context.
shuffle' :: MonadRandom m => [a] -> m [a]
shuffle' xs = do
  ns <- getRandomRs (0, length xs - 1)
  return $ map (xs !!) ns

normalize :: [Float] -> [Float]
normalize xs =
  let maxVal = maximum xs
      minVal = minimum xs
   in map (\x -> (x - minVal) / (maxVal - minVal)) xs


activationFunctions :: ActivationType -> (Float -> Float, Float -> Float)
activationFunctions Sigmoid = (sigmoid, sigmoid')
activationFunctions Relu = (reluFunction, reluDerivative)