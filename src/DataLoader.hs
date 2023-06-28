module DataLoader where

import Data.Binary.Get ( getWord8, Get, getWord32be, runGet )
import Data.ByteString.Lazy qualified as BL
import Data.Vector.Storable qualified as VS
import Utility (normalize, oneHotEncode) -- Import the functions from Utility.hs

type Image = VS.Vector Float
type Label = Int
type Inputs = [Float]
type Target = [Float]

readImageHeader :: BL.ByteString -> (Int, Int, Int, Int)
readImageHeader = runGet getHeader
  where
    getHeader = do
      magicNumber <- getWord32be -- Magic number
      numImages <- getWord32be -- Number of images
      numRows <- getWord32be -- Number of rows
      numColumns <- getWord32be -- Number of columns
      return (fromIntegral magicNumber, fromIntegral numImages, fromIntegral numRows, fromIntegral numColumns)

readLabelHeader :: BL.ByteString -> (Int, Int)
readLabelHeader = runGet getHeader
  where
    getHeader = do
      magicNumber <- getWord32be -- Magic number
      numLabels <- getWord32be -- Number of labels
      return (fromIntegral magicNumber, fromIntegral numLabels)

readImage :: Get Image
readImage = VS.map ((/ 255) . fromIntegral) . VS.fromList <$> replicateM (28 * 28) getWord8

readLabel :: Get Label
readLabel = fromIntegral <$> getWord8

loadImages :: FilePath -> IO [Image]
loadImages path = do
  input <- BL.readFile path
  let (_, numImages, _, _) = readImageHeader input
      images = runGet (replicateM numImages readImage) (BL.drop 16 input)
  return images

loadLabels :: FilePath -> IO [Label]
loadLabels path = do
  input <- BL.readFile path
  let (_, numLabels) = readLabelHeader input
      labels = runGet (replicateM numLabels readLabel) (BL.drop 8 input)
  return labels

loadData :: FilePath -> FilePath -> IO [(Image, Label)]
loadData imagesPath labelsPath = do
  images <- loadImages imagesPath
  labels <- loadLabels labelsPath
  return (zip images labels)

normalizeImage :: Image -> Image
normalizeImage = VS.map ((* 2) . subtract 0.5) -- maps values from [0, 1] to [-1, 1]

toInputs :: Image -> Inputs
toInputs = VS.toList

preprocess :: [(Image, Label)] -> [(Inputs, Target)]
preprocess = map (bimap (normalize . toInputs) oneHotEncode)

loadTrainingData :: IO [(Inputs, Target)]
loadTrainingData = preprocess <$> loadData "Data/train-images.idx3-ubyte" "Data/train-labels.idx1-ubyte"

loadValidationData :: IO [(Inputs, Target)]
loadValidationData = preprocess <$> loadData "Data/t10k-images.idx3-ubyte" "Data/t10k-labels.idx1-ubyte"