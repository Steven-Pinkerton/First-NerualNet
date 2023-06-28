module Network (
  initializeNetwork,
  calculateNetworkOutputs,
) where

import Data.List (zipWith3)
import Forward (calculateLayerOutputs)
import System.Random (newStdGen, randoms)
import Types (ActivationType, Layer, Network, Neuron (..))

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
