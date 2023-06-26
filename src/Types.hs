{-# LANGUAGE InstanceSigs #-}
module Types where

import Text.Show (show)

-- The ActivationFunction type is a tuple of two functions:
-- the activation function itself and its derivative.
-- This is used in the forward pass (to calculate the output of a neuron)
-- and in the backward pass (to calculate the gradients).
type ActivationFunction = (Float -> Float, Float -> Float) -- (function, derivative)

-- The Neuron type represents a single neuron in the network.
-- It has a list of weights (one for each input), a bias, and an activation function.
data Neuron = Neuron
  { weights :: [Float]
  , bias :: Float
  , activation :: ActivationFunction
  }

-- Custom Show instance for Neuron for prettier printing
instance Show Neuron where
  show :: Neuron -> String
  show (Neuron weights' bias' _) =
    "Neuron {weights = " ++ Text.Show.show weights' ++ ", bias = " ++ Text.Show.show bias' ++ "}"

-- A Layer is simply a list of Neurons.
-- All neurons in a layer have the same number of weights,
-- corresponding to the number of neurons in the previous layer (or the number of inputs if it's the first layer).
type Layer = [Neuron]

-- A Network is a list of Layers.
-- The output of each layer is used as the input for the next layer.
type Network = [Layer]

-- Inputs to the network is a list of floats. For the MNIST task, this would be a 784-element list
-- representing the 28x28 pixel intensities of the input image.
type Inputs = [Float] -- length should be 784

-- The Target for the MNIST task is represented as an Int in the range 0..9,
-- but this needs to be converted to a one-hot encoded list of 10 elements for training the network.
type Target = Int -- should be in the range [0..9]

-- The Outputs from the network is a list of 10 probabilities, one for each digit from 0 to 9.
-- The index of the maximum element in this list is the digit the network is predicting.
type Outputs = [Float] -- length should be 10

-- LearningRate defines how much we want to update our weights and biases in each training step.
-- This is a hyperparameter that you can tweak. Too high learning rate might cause the model
-- to converge too quickly to a suboptimal solution, whereas too low learning rate might cause
-- the model to get stuck or to converge too slowly.
type LearningRate = Float

-- Epochs is the number of times the learning algorithm will work through the entire training dataset.
-- Training a neural network for more epochs might lead to better performance but also risk overfitting.
type Epochs = Int

-- BatchSize is the number of training examples utilized in one iteration.
-- It's a compromise between faster training (larger batch size) and better convergence (smaller batch size).
type BatchSize = Int