module Loss where
import GHC.Float (log)


-- Cross entropy loss for a single example
-- This function is used during the training of the network to quantify the difference
-- between the predicted probabilities (yPred) and the true classes (yTrue).
-- yTrue is the one-hot encoded true class vector. For example, if the true class is "3",
-- then yTrue should be [0, 0, 0, 1, 0, 0, 0, 0, 0, 0].
-- yPred is the predicted probabilities for each class, as output by the network. For example,
-- it could be something like [0.1, 0.05, 0.1, 0.6, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1].
-- The output is a single Float value representing the cross entropy loss. Lower values mean
-- that the predicted probabilities are closer to the true classes, and therefore that the
-- network is performing better.
crossEntropyLoss :: [Float] -> [Float] -> Float
crossEntropyLoss yTrue yPred = - sum (zipWith (\yT yP -> yT * log yP) yTrue yPred)