module ModelIO where

import Data.ByteString qualified as BS
import Data.Serialize (decode, encode)
import Types ( Network )

-- | Save a neural network model to a file.
saveModel :: FilePath -> Network -> IO ()
saveModel path network = BS.writeFile path (encode network)

-- | Load a neural network model from a file.
loadModel :: FilePath -> IO (Either String Network)
loadModel path = do
  modelData <- BS.readFile path
  return $ decode modelData