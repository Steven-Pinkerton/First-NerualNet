{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}

{-# HLINT ignore "Use readFileBS" #-}
module ModelIO where

import Data.Binary (decode, encode)
import Types (Network)

-- | Save a neural network model to a file.
saveModel :: FilePath -> Network -> IO ()
saveModel filePath network = writeFileLBS filePath (encode network)

-- | Load a neural network model from a file.
loadModel :: FilePath -> IO (Either String Network)
loadModel path = do
  modelData <- readFileLBS path
  return $ decode modelData
