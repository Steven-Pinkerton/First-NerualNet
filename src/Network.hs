module Network 
( initializeNetwork
, calculateNetworkOutputs
  ) where

import System.Random (newStdGen, randoms)
import Types (Layer, Network, Neuron (..), ActivationType)
import Forward ( calculateLayerOutputs )
import Data.List ( zipWith3 )



calculateNetworkOutputs :: Network -> [Float] -> Maybe [[Float]]
calculateNetworkOutputs network inputs = Just $ scanl (flip calculateLayerOutputs) inputs network

initializeNetwork :: ActivationType -> [Int] -> IO Network
initializeNetwork activationFunc architecture = do
  gen <- newStdGen
  let networkWeights = map (\n -> take n $ randoms gen :: [Float]) architecture
  let networkBiases = randoms gen :: [Float]
  return $ zipWith3 createLayer networkWeights networkBiases architecture
  where
    createLayer :: [Float] -> Float -> Int -> Layer
    createLayer weights' bias' neurons = replicate neurons (Neuron weights' bias' activationFunc)