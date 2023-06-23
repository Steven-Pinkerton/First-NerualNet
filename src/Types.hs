module Types where
import Text.Show (show)

-- Define the Activation Function type
type ActivationFunction = (Float -> Float, Float -> Float) -- (function, derivative)

-- Define the Neuron type
data Neuron = Neuron
  { weights :: [Float]
  , bias :: Float
  , activation :: ActivationFunction
  }

-- Custom Show instance for Neuron
instance Show Neuron where
  show (Neuron weights' bias' _) =
    "Neuron {weights = " ++ Text.Show.show weights' ++ ", bias = " ++ Text.Show.show bias' ++ "}"

-- Define the Layer type
type Layer = [Neuron]

-- Define the Network type
type Network = [Layer]