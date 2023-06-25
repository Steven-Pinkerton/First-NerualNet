module Utility where
import Types ( Layer, Neuron(weights), ActivationFunction )
import System.Random ( RandomGen, Random (randomR) )
import Data.List ( (!!) )


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
shuffle :: RandomGen g => [a] -> g -> ([a], g)
shuffle [] g = ([], g)
shuffle xs g =
  let (n, g') = randomR (0, length xs - 1) g
      front = take n xs
      back = drop (n + 1) xs
   in let (shuffledRest, g'') = shuffle (front ++ back) g'
       in ((xs !! n) : shuffledRest, g'')