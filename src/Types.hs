{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}


module Types where

import Data.Serialize (Serialize)
import Data.Binary ( Binary )

-- The ActivationType is just a simple enumeration of the
-- possible activation functions that a neuron could use.
data ActivationType = Sigmoid | ReLU
  deriving stock (Generic, Show)
  deriving anyclass (Serialize, Binary)

-- The Neuron type represents a single neuron in the network.
-- It has a list of weights (one for each input), a bias, and an activation type.
data Neuron = Neuron
  { weights :: [Float]
  , bias :: Float
  , activationType :: ActivationType
  }
  deriving stock (Generic, Show)
  deriving anyclass (Serialize, Binary)

-- A Layer is simply a list of Neurons.
-- All neurons in a layer have the same number of weights,
-- corresponding to the number of neurons in the previous layer (or the number of inputs if it's the first layer).
type Layer = [Neuron]

-- The original class label, which is an Int in the range 0..9.
type Label = Int

-- A Network is a list of Layers.
-- The output of each layer is used as the input for the next layer.
type Network = [Layer]

-- Inputs to the network is a list of floats. For the MNIST task, this would be a 784-element list
-- representing the 28x28 pixel intensities of the input image.
type Inputs = [Float] -- length should be 784

-- The Target for the MNIST task is represented as an Int in the range 0..9,
-- but this needs to be converted to a one-hot encoded list of 10 elements for training the network.
type Target = [Float] -- should be in the range [0..9]

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

-- The Delta represents the error calculated during backpropagation.
-- This is typically a list of Floats, depending on the structure of your network.
type Delta = [Float] -- If Delta is represented as a list of floats
