module NeuralLayerDense (initialiseLayer
                        , LayerDense(..)
                        , activatedLayerOutputsForBatch
                        , derivativeReluBatch
                        , derivativeRelu
                        , layerOutputsForBatch
                        , activationRelu
                        , dotProductForBatch
                        , weightedInputsForBatch
                        , clipArray
                        , mmult
                        , dotProduct
                        , categoricalCrossEntropyLossBatched
                        , oneHotVectors
                        , activatedLayerOutput
                        , mean
                        , accuracy
                        , activationSoftMax) where

import System.Random
import Control.Monad
import Data.Ord
import Data.List

data LayerDense = LayerDense {
    inputs :: [[Double]]
  , weights :: [[Double]]
  , biases :: [Double]
  , dWeights :: [[Double]]
  , dBiases :: [Double]
  , dInputs :: [[Double]]
}

initialiseLayer :: [[Double]] -> Int -> IO LayerDense
initialiseLayer is neuronCount = do
  let inputCountPerNeuron = length (head is)
  randomWeights <- generateRandomWeights inputCountPerNeuron neuronCount

  return LayerDense {
       inputs = is
     , weights = randomWeights
     , biases = replicate neuronCount 0
     , dWeights = []
     , dBiases = []
     , dInputs = []
  }

backwardPass :: LayerDense -> [[Double]] -> LayerDense
backwardPass l dValues = LayerDense {
       inputs = inputs l
     , weights = weights l
     , biases = biases l
     , dWeights = mmult (transpose (inputs l)) dValues
     , dBiases = map sum (transpose dValues)
     , dInputs = mmult dValues (weights l)
  }

generateRandomWeights :: Int -> Int -> IO [[Double]]
generateRandomWeights inputCountPerNeuron neuronCount = Control.Monad.replicateM neuronCount (randomList inputCountPerNeuron)

randomList :: Int -> IO [Double]
randomList n = Control.Monad.replicateM n randomNumber

randomNumber :: IO Double
randomNumber = do
  r <- randomRIO (-1.0,1.0)
  return (0.1 * r)

dotProduct :: [Double] -> [Double] -> [Double]
dotProduct [x] [y] = [x * y]
dotProduct xs ys = (head xs * head ys) : dotProduct (tail xs) (tail ys)

dotProductForBatch :: [[Double]] -> [[Double]] -> [[Double]]
dotProductForBatch [x] [y] = [dotProduct x y]
dotProductForBatch xs ys = dotProduct (head xs) (head ys) : dotProductForBatch (tail xs) (tail ys)

neuronOutput :: [Double] -> [Double] -> Double -> Double
neuronOutput is ws = (+) (weightedInputs is ws)

layerOutput :: [Double] -> [[Double]] -> [Double] -> [Double]
layerOutput is [w] [b] = [neuronOutput is w b]
layerOutput is ws bs = neuronOutput is (head ws) (head bs) : layerOutput is (tail ws) (tail bs)

layerOutputsForBatch :: [[Double]] -> [[Double]] -> [Double] -> [[Double]]
layerOutputsForBatch [i] ws bs = [layerOutput i ws bs]
layerOutputsForBatch is ws bs = layerOutput (head is) ws bs : layerOutputsForBatch (tail is) ws bs

activationRelu :: [Double] -> [Double]
activationRelu = map (\x -> if x > 0 then x else 0)

activationSoftMax :: [Double] -> [Double]
activationSoftMax xs = normalise $ exponentiate xs

exponentiate :: [Double] -> [Double]
exponentiate = map (exp 1 **)

normalise :: [Double] -> [Double]
normalise xs = map (\x -> (/) x (normBase xs)) xs

normBase :: [Double] -> Double
normBase = sum

activatedLayerOutput :: [Double] -> [[Double]] -> [Double] -> ([Double] -> [Double]) -> [Double]
activatedLayerOutput is ws bs af = af (layerOutput is ws bs)

activatedLayerOutputsForBatch :: [[Double]] -> [[Double]] -> [Double] -> ([Double] -> [Double]) -> [[Double]]
activatedLayerOutputsForBatch [i] ws bs af = [activatedLayerOutput i ws bs af]
activatedLayerOutputsForBatch is ws bs af = activatedLayerOutput (head is) ws bs af : activatedLayerOutputsForBatch (tail is) ws bs af

clip :: Double -> Double
clip x
  | x < -1e-7 = -1e-7
  | x > (1 - 1e-7) = 1 - 1e-7
  | otherwise = x

clipArray :: [Double] -> [Double]
clipArray = map clip

weightedInputs :: [Double] -> [Double] -> Double
weightedInputs [i] [w] = i * w
weightedInputs is ws = sum [ x * y | (x, y) <- zip is ws ]

weightedInputsForBatch :: [[Double]] -> [[Double]] -> [Double]
weightedInputsForBatch [i] [w] = [weightedInputs i w]
weightedInputsForBatch is ws = weightedInputs (head is) (head ws) : weightedInputsForBatch (tail is) (tail ws)

logSourceTimesTarget :: [Double] -> [Int] -> Double
logSourceTimesTarget [a] [t] = log a * fromIntegral t
logSourceTimesTarget actual target = log (head actual) * fromIntegral (head target) + logSourceTimesTarget (tail actual) (tail target)

oneHotVector :: Int -> Int -> [Int]
oneHotVector hotPosition size = replicate hotPosition 0 ++ [1] ++ replicate (size - hotPosition - 1) 0

oneHotVectors :: [Int] -> Int -> [[Int]]
oneHotVectors positions size = map (`oneHotVector` size) positions

categoricalCrossEntropyLoss :: [Double] -> [Int] -> Double
categoricalCrossEntropyLoss actual target = negate (logSourceTimesTarget actual target)

categoricalCrossEntropyLossBatched :: [[Double]] -> [[Int]] -> [Double]
categoricalCrossEntropyLossBatched [a] [t] = [categoricalCrossEntropyLoss a t]
categoricalCrossEntropyLossBatched actuals targets =
  categoricalCrossEntropyLoss (head actuals) (head targets) : categoricalCrossEntropyLossBatched (tail actuals) (tail targets)

listDivide :: [Double] -> [Int] -> Double
listDivide [a] [t] = fromIntegral t / a
listDivide actual target = fromIntegral (head target) / head actual + listDivide (tail actual) (tail target)

categoricalCrossEntropyLossBackward :: [Double] -> [Int] -> Double
categoricalCrossEntropyLossBackward dValues yTrueOneVector = negate (listDivide dValues yTrueOneVector) / fromIntegral (length dValues)

categoricalCrossEntropyLossBatchedBackward :: [[Double]] -> [[Int]] -> [Double]
categoricalCrossEntropyLossBatchedBackward [a] [t] = [categoricalCrossEntropyLossBackward a t]
categoricalCrossEntropyLossBatchedBackward actuals targets =
  categoricalCrossEntropyLossBackward (head actuals) (head targets) : categoricalCrossEntropyLossBatchedBackward (tail actuals) (tail targets)
  
mean :: [Double] -> Double
mean xs = sum xs / fromIntegral (length xs)

maxWIndex :: (Ord a1, Ord a2, Num a2, Enum a2) => [a1] -> (a2, a1)
maxWIndex = maximumBy (comparing snd <> flip (comparing fst)) . zip [0..]

indexOfLargestValue :: [Double] -> Int
indexOfLargestValue xs = fst (maxWIndex xs)

argmax :: [Double] -> [Int]
argmax xs = oneHotVector (indexOfLargestValue xs) (length xs)

isAccurateOneHot :: [Double] -> [Int] -> Int
isAccurateOneHot prediction hotVector = if indexOfLargestValue prediction == indexOfLargestValue (map fromIntegral hotVector) then 1 else 0

totalAccurateOneHot :: [[Double]] -> [[Int]] -> Int
totalAccurateOneHot [x] [y] = isAccurateOneHot x y
totalAccurateOneHot xs ys = isAccurateOneHot (head xs) (head ys) + totalAccurateOneHot (tail xs) (tail ys)

accuracyFromOneHotVectors :: [[Double]] -> [[Int]] -> Double
accuracyFromOneHotVectors xs ys = fromIntegral (totalAccurateOneHot xs ys) / fromIntegral (length xs)

isAccurate :: [Double] -> Int -> Int
isAccurate prediction classNum = isAccurateOneHot prediction (oneHotVector classNum (length prediction))

totalAccurate :: [[Double]] -> [Int] -> Int
totalAccurate [x] [y] = isAccurate x y
totalAccurate predictions classNums = isAccurate (head predictions) (head classNums) + totalAccurate (tail predictions) (tail classNums)

accuracy :: [[Double]] -> [Int] -> Double
accuracy xs ys = fromIntegral (totalAccurate xs ys) / fromIntegral (length xs)

derivativeRelu :: [Double] -> [Double] -> [Double]
derivativeRelu [x] [y] = if x <= 0 then [0] else [y]
derivativeRelu xs ys = (if head xs <= 0 then 0.0 else head ys) : derivativeRelu (tail xs) (tail ys)

derivativeReluBatch :: [[Double]] -> [[Double]] -> [[Double]]
derivativeReluBatch [x] [y] = [derivativeRelu x y]
derivativeReluBatch xs ys = derivativeRelu (head xs) (head ys) : derivativeReluBatch (tail xs) (tail ys)

mmult :: Num a => [[a]] -> [[a]] -> [[a]]
mmult a b = [ [ sum $ zipWith (*) ar bc | bc <- transpose b ] | ar <- a ]

