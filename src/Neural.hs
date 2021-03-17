{-# LANGUAGE ExtendedDefaultRules #-}

module Neural where

import NeuralData
import NeuralLayerDense
import Graphics.Matplotlib
import Data.List

f :: Double -> Double
f x = 2 * x ** 2

tangentLine :: Double -> Double -> Double -> Double
tangentLine ad b x = ad * x + b

zippy2 :: Double -> Double -> Double
zippy2 a b = a + b * (-0.001)

zippy :: [Double] -> [Double] -> [Double]
zippy = zipWith zippy2

go :: IO ()
go = do
--  let x = [0,0.001..5]
--  let y = map f x
--
--  let p2Delta = 0.0001
--  let x1 = 2
--  let x2 = x1 + p2Delta
--
--  let y1 = f x1
--  let y2 = f x2
--
--  let approxDerivative = (y2-y1) / (x2-x1)
--  let b = y2 - approxDerivative * x2
--  let toPlot = [x1-0.9,x1,x1+0.9]
--
--  onscreen $ lineF f x
--  onscreen $ lineF (tangentLine approxDerivative b) toPlot
--  print $ map (tangentLine approxDerivative b) toPlot

--  3 inputs to a single neuron, a list of one list of 3 weights (1 set) and a list biases (1)
--  y = ReLU(sum(mul(x0, w0), mul(x1, w1), mul(x2, w2), b))
--  print $ activatedLayerOutput [1.0,-2.0,3.0] [[-3.0,-1.0,2.0]] [1.0] activationRelu
--
--  let dvalues = [[1.0, 1.0, 1.0],[2.0, 2.0, 2.0],[3.0, 3.0, 3.0]]
--  let inputs = [[1, 2, 3, 2.5],[2.0, 5.0, -1.0, 2],[-1.5, 2.7, 3.3, -0.8]]
--  let weights = [[0.2, 0.8, -0.5, 1],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
--  let biases = [[2, 3, 0.5]]
--  let lofb = layerOutputsForBatch inputs weights (head biases)
--  let reluOutputs = map activationRelu lofb
--  let dRelu = derivativeReluBatch reluOutputs lofb
--  let dInputs = mmult dRelu weights
--  let dWeights = mmult (transpose inputs) dRelu
--  let dBiases = [map sum (transpose dRelu)]
--  let nWeights = zipWith zippy weights dWeights
--  let nBiases = zipWith zippy biases dBiases
--
--  print dRelu
--  print weights
--  print dInputs
--  print dWeights
--  print dBiases
--  print nWeights
--  print nBiases

  let spiralList = listOfLists spiralData
  --onscreen $ scatter (spiralXs spiralData) (spiralYs spiralData)
  dense1 <- initialiseLayer spiralList 3
  let layer1Outputs = map activationRelu $ layerOutputsForBatch (NeuralLayerDense.inputs dense1) (weights dense1) (biases dense1)
  dense2 <- initialiseLayer layer1Outputs 3

  let layer2Outputs = map activationSoftMax $ layerOutputsForBatch layer1Outputs (weights dense2) (biases dense2)
  let categoryVectors = oneHotVectors spiralClasses 3
  let clippedLayer2Outputs = map clipArray layer2Outputs
  let loss = mean $ categoricalCrossEntropyLossBatched clippedLayer2Outputs categoryVectors
  let acc = accuracy layer2Outputs spiralClasses

  print loss
  print acc


