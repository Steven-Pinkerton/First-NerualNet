module Utility where
import Types ( Layer, Neuron(weights), ActivationFunction )

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