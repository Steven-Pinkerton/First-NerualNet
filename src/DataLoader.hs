{-# LANGUAGE OverloadedStrings #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use 'fromMaybe' from Relude" #-}
{-# HLINT ignore "Use 'bimap' from Relude" #-}
{-# HLINT ignore "Use 'lines' from Relude" #-}
{-# HLINT ignore "Use readFileText" #-}

module DataLoader where

import Data.Binary.Get qualified as G
import Data.ByteString.Lazy qualified as BL
import Data.Text qualified as T
import Data.Text.IO qualified as TIO
import Types (Inputs, Target, Label)
import Data.Bifunctor (bimap)
import Data.Maybe (fromMaybe)
import Data.Text.Read (double)
import Utility ( oneHotEncode, normalize )


readImageHeader :: BL.ByteString -> (Int, Int, Int, Int)
readImageHeader = G.runGet getHeader
  where
    getHeader = do
      magicNumber <- G.getWord32be -- Magic number
      numImages <- G.getWord32be -- Number of images
      numRows <- G.getWord32be -- Number of rows
      numColumns <- G.getWord32be -- Number of columns
      return (fromIntegral magicNumber, fromIntegral numImages, fromIntegral numRows, fromIntegral numColumns)

readLabelHeader :: BL.ByteString -> (Int, Int)
readLabelHeader = G.runGet getHeader
  where
    getHeader = do
      magicNumber <- G.getWord32be -- Magic number
      numLabels <- G.getWord32be -- Number of labels
      return (fromIntegral magicNumber, fromIntegral numLabels)


-- | Load the training data from a file
loadTrainingData :: FilePath -> IO [(Inputs, Label)]
loadTrainingData path = do
  rawData <- TIO.readFile path
  return $ map parseDataLine $ T.lines rawData

-- | Load the validation data from a file
loadValidationData :: FilePath -> IO [(Inputs, Label)]
loadValidationData path = do
  rawData <- TIO.readFile path
  return $ map parseDataLine $ T.lines rawData

-- | Preprocess the data
preprocess :: [(Inputs, Label)] -> [(Inputs, Target)]
preprocess = map (bimap normalize oneHotEncode)


parseDataLine :: T.Text -> (Inputs, Label)
parseDataLine line =
  let numbers = map parseToFloat $ T.splitOn "," line
   in (fromMaybe [] $ viaNonEmpty tail numbers, round . fromMaybe 0 $ viaNonEmpty head numbers)
  where
    parseToFloat t = case double t of
      Right (num, _) -> realToFrac num :: Float
      Left _ -> 0