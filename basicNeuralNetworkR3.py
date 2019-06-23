#Originally created 2/14/19 by QPU Misaligned 
#Release 3 created 6/22/19 by QPU Misaligned
#https://github.com/QPU-Misaligned/Basic-Neural-Network

import math
import random

class basicNeuralNet:
  #__init__ method
  def __init__(self, nodesPerLayer, learningRate):
    self.nodesPerLayer = nodesPerLayer
    self.learningRate = learningRate
    self.activationFunction = self.ELU
    self.dx = 0.0000000001
    weights = []
    for i in range(0, len(nodesPerLayer)-1):
      layerWeights = []
      for j in range(0, nodesPerLayer[i+1]):
        nodeWeights = []
        for k in range(0, nodesPerLayer[i]):
          nodeWeights.append(random.random())
        layerWeights.append(nodeWeights)
      weights.append(layerWeights)
    biases = []
    for i in range(0, len(nodesPerLayer)-1):
      layerBiases = []
      for j in range(0, nodesPerLayer[i+1]):
        layerBiases.append([random.random()])
      biases.append(layerBiases)
    self.weightsBiases = [weights, biases]
    weightsV = []
    for i in range(0, len(nodesPerLayer)-1):
      layerWeightsV = []
      for j in range(0, nodesPerLayer[i+1]):
        nodeWeightsV = []
        for k in range(0, nodesPerLayer[i]):
          nodeWeightsV.append(0)
        layerWeightsV.append(nodeWeightsV)
      weightsV.append(layerWeightsV)
    biasesV = []
    for i in range(0, len(nodesPerLayer)-1):
      layerBiasesV = []
      for j in range(0, nodesPerLayer[i+1]):
        layerBiasesV.append([0])
      biasesV.append(layerBiases)
    self.weightsBiasesV = [weightsV, biasesV]
  
  #getters and setters
  def getNodesPerLayer(self):
    return self.nodesPerLayer
  
  def getLearningRate(self):
    return self.learningRate
  
  def setLearningRate(self, learningRate):
    self.learningRate = learningRate
  
  def getWeightsBiases(self):
    return self.weightsBiases
  
  def setWeightsBiases(self, weightsBiases):
    self.weightsBiases = weightsBiases
  
  def getActivationFunction(self):
    return self.activationFunction
  
  def setActivationFunction(self, activationFunction):
    self.activationFunction = activationFunction
  
  #getters and setters for individual weights/biases
  def getWeightBias(self, weightsBiases, location):
    output = []
    subArray = weightsBiases[:]
    for i in location:
      output.append(subArray)
      subArray = subArray[i]
    output.append(subArray)
    return output
  
  def setWeightBias(self, weightsBiases, location, inputValue):
    subArrays = self.getWeightBias(weightsBiases, location)
    subArrays[-1] = inputValue
    for i in range(0, len(subArrays)-1):
      subArrays[len(subArrays)-i-2][location[len(subArrays)-i-2]] = subArrays[len(subArrays)-i-1]
    weightsBiases = subArrays[0]
    return subArrays
  
  #activation functions
  def linear(self, inputValue):
    return inputValue
  
  def sigmoid(self, inputValue):
    return 1/(1+math.exp(-1*inputValue))
  
  def tanh(self, inputValue):
    return (math.exp(inputValue)-math.exp(-1*inputValue))/(math.exp(inputValue)+math.exp(-1*inputValue))
  
  def ReLU(self, inputValue):
    if(inputValue>0):
      return inputValue
    return 0
  
  def LReLU(self, inputValue):
    if(inputValue>0):
      return inputValue
    return 0.01*inputValue
  
  def ELU(self, inputValue):
    if(inputValue>0):
      return inputValue
    return math.exp(inputValue)-1
  
  def SReLU(self, inputValue):
    return math.log(1+math.exp(inputValue))
  
  def SLReLU(self, inputValue):
    return 0.01*inputValue+(0.99)*math.log(1+math.exp(inputValue))
  
  def SELU(self, inputValue):
    return math.log(1+math.exp(inputValue+1))-1
  
  #node output and network output methods
  def nodeOutputA(self, inputValues, inputWeights, inputBias, activationFunction):
    aggregate = inputBias
    for i in range(0, len(inputValues)):
      aggregate += inputValues[i]*inputWeights[i]
    return activationFunction(aggregate)
  
  def nodeOutput(self, inputValues, inputWeights, inputBias):
    return self.nodeOutputA(inputValues, inputWeights, inputBias, self.activationFunction)
  
  def networkOutputA(self, inputValues, activationFunction):
    layerInput = inputValues[:]
    for i in range(0, len(self.nodesPerLayer)-1):
      layerOutput = []
      for j in range(0, self.nodesPerLayer[i+1]):
        layerOutput.append(self.nodeOutputA(layerInput, self.weightsBiases[0][i][j], self.weightsBiases[1][i][j][0], activationFunction))
      layerInput = layerOutput[:]
    return layerOutput
  
  def networkOutput(self, inputValues):
    return self.networkOutputA(inputValues, self.activationFunction)
  
  #cost function for a dataset
  def costA(self, dataSet, activationFunction):
    sum = 0
    for dataPoint in dataSet:
      networkOutput = self.networkOutputA(dataPoint[0], activationFunction)
      for i in range(0, len(networkOutput)):
        sum += (networkOutput[i]-dataPoint[1][i])**2
    return sum
  
  def cost(self, dataSet):
    return self.costA(dataSet, self.activationFunction)
  
  #gradient finding function for a dataset
  def gradientA(self, weightsBiases, dataSet, activationFunction):
    gradient = []
    costZero = self.costA(dataSet, activationFunction)
    for weightOrBias in range(0, len(self.weightsBiases)):
      for layer in range(0, len(self.weightsBiases[weightOrBias])):
        for node in range(0, len(self.weightsBiases[weightOrBias][layer])):
          for weightBiasValue in range(0, len(self.weightsBiases[weightOrBias][layer][node])):
            weightBiasZero = self.weightsBiases[weightOrBias][layer][node][weightBiasValue]
            self.setWeightBias(weightsBiases, [weightOrBias, layer, node, weightBiasValue], weightBiasZero+self.dx)
            gradient.append((self.costA(dataSet, activationFunction)-costZero)/self.dx)
            self.setWeightBias(weightsBiases, [weightOrBias, layer, node, weightBiasValue], weightBiasZero)
    return gradient
  
  def gradient(self, dataSet):
    return self.gradientA(self. weightsBiases, dataSet, self.activationFunction)
  
  #optimize with stochastic gradient descent function for a dataset 1st order
  def optimizeSGDA(self, weightsBiases, dataSet, activationFunction):
    g = self.gradientA(weightsBiases, dataSet, activationFunction)
    i = 0
    for weightOrBias in range(0, len(weightsBiases)):
      for layer in range(0, len(weightsBiases[weightOrBias])):
        for node in range(0, len(weightsBiases[weightOrBias][layer])):
          for weightBiasValue in range(0, len(weightsBiases[weightOrBias][layer][node])):
            self.setWeightBias(weightsBiases, [weightOrBias, layer, node, weightBiasValue], weightsBiases[weightOrBias][layer][node][weightBiasValue]-self.learningRate*g[i])
            i += 1
    return self.costA(dataSet, activationFunction)
  
  def optimizeSGD(self, dataSet):
    return self.optimizeSGDA(self.weightsBiases, dataSet, self.activationFunction)
  
  #optimize with momentum for a dataset 2nd order
  def optimizeMomentumA(self, weightsBiasesV, weightsBiases, dataSet, activationFunction, friction):
    g = self.gradientA(weightsBiases, dataSet, activationFunction)
    i = 0
    for weightOrBias in range(0, len(weightsBiasesV)):
      for layer in range(0, len(weightsBiasesV[weightOrBias])):
        for node in range(0, len(weightsBiasesV[weightOrBias][layer])):
          for weightBiasValue in range(0, len(weightsBiasesV[weightOrBias][layer][node])):
            self.setWeightBias(weightsBiasesV, [weightOrBias, layer, node, weightBiasValue], (1-friction)*weightsBiasesV[weightOrBias][layer][node][weightBiasValue]+self.learningRate*g[i])
            self.setWeightBias(weightsBiases, [weightOrBias, layer, node, weightBiasValue], weightsBiases[weightOrBias][layer][node][weightBiasValue]-self.learningRate*weightsBiasesV[weightOrBias][layer][node][weightBiasValue])
            i+=1
    return self.costA(dataSet, activationFunction)
  
  def optimizeMomentum(self, dataSet, friction):
    return self.optimizeMomentumA(self.weightsBiasesV, self.weightsBiases, dataSet, self.activationFunction, friction)
