{-# LANGUAGE OverloadedStrings #-}
module DataLoader where
import Types ( Target, Inputs )
import Data.Text (pack)

-- | Load the training data from a file
loadTrainingData :: FilePath -> IO [(Inputs, Target)]
loadTrainingData path = do
  rawData <- readFile path
  return $ map parseDataLine $ lines (pack rawData)


-- | Load the validation data from a file
loadValidationData :: FilePath -> IO [(Inputs, Target)]
loadValidationData path = do
  rawData <- readFile path
  return $ map parseDataLine $ lines (pack rawData)

-- | Preprocess the data
preprocess :: [(Inputs, Target)] -> [(Inputs, Target)]
preprocess = map (\(inputs, target) -> (normalize inputs, oneHotEncode target))

parseDataLine :: String -> (Inputs, Target)
parseDataLine line =
  let numbers = map read $ splitOn "," line :: [Float]
   in (tail numbers, round $ head numbers)