module Backward where


-- Compute the error of a single neuron in the output layer
calculateOutputError :: Float -> Float -> Float -> Float
calculateOutputError target output outputDerivative =
  -- The error is the derivative of the loss function with respect to the output times the derivative of the output
  (output - target) * outputDerivative

-- Compute the delta of a single neuron in the output layer
calculateOutputDelta :: Float -> Float
calculateOutputDelta error' =
  -- The delta is the error times the derivative of the activation function
  error' * 1

-- Compute the error of a single neuron in a hidden layer
calculateHiddenError :: [Float] -> [Float] -> Float -> Float
calculateHiddenError nextWeights nextDeltas activationDerivative =
  -- The error is the sum of the weights times deltas of the next layer times the derivative of the output
  sum (zipWith (*) nextWeights nextDeltas) * activationDerivative

-- Compute the delta of a single neuron in a hidden layer
calculateHiddenDelta :: Float -> Float
calculateHiddenDelta error' = error' * 1

-- Update a single weight
updateWeight :: Float -> Float -> Float -> Float -> Float
updateWeight weight learningRate delta neuronOutput =
  -- The new weight is the old weight minus the learning rate times the delta times the output of the neuron
  weight - learningRate * delta * neuronOutput

-- Update a single bias
updateBias :: Float -> Float -> Float -> Float
updateBias bias learningRate delta =
  -- The new bias is the old bias minus the learning rate times the delta
  bias - learningRate * delta