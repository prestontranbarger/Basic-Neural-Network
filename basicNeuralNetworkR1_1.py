#Originally created 2/14/19 by QPU Misaligned
#Release 1.1 created 2/17/19 by QPU Misaligned 

import math
import random

class basicNeuralNet:
  #__init__ method
  def __init__(self, nodes, learningRate):
    self.nodes = nodes
    self.learningRate = learningRate
    self.accuracy = 0.0000000001
    weights = []
    for i in range(0,len(nodes)-1):
      layerWeights = []
      for j in range(0,nodes[i+1]):
        nodeWeights = []
        for k in range(0,nodes[i]):
          nodeWeights.append(random.random())
        layerWeights.append(nodeWeights)
      weights.append(layerWeights)
    self.weights = weights
  
  #getters and setters
  def getNodes(self):
    return self.nodes
  
  def getLearningRate(self):
    return self.learningRate
  
  def setLearningRate(self, value):
    self.learningRate = value
  
  def getWeights(self):
    return self.weights

  #weight getter and setter
  def getWeight(self, location):
    output = []
    subArray = self.weights[:]
    for i in location:
      output.append(subArray)
      subArray = subArray[i]
    output.append(subArray)
    return output

  def setWeight(self, location, value):
    subArrays = self.getWeight(location)
    subArrays[-1] = value
    for i in range(0,len(subArrays)-1):
      subArrays[len(subArrays)-i-2][location[len(subArrays)-i-2]] = subArrays[len(subArrays)-i-1]
    self.weights = subArrays[0]
    return subArrays

  #node output and network output methods
  def nodeOutput(self, inputValues, inputWeights):
    aggregate = 0
    for i in range(0,len(inputValues)):
      aggregate += inputValues[i]*inputWeights[i]
    return 1/(1+math.exp(-1*aggregate))

  def networkOutput(self, inputValues):
    layerInput = inputValues[:]
    for i in range(0,len(self.nodes)-1):
      layerOutput = []
      for j in range(0,self.nodes[i+1]):
        layerOutput.append(self.nodeOutput(layerInput,self.weights[i][j]))
      layerInput = layerOutput[:]
    return layerOutput

  #cost function for a dataset
  def cost(self, dataSet):
    sum = 0
    for dataPoint in dataSet:
      netOut = self.networkOutput(dataPoint[0])
      for i in range(0,len(netOut)):
        sum += (netOut[i]-dataPoint[1][i])**2
    return sum

  #gradient finding function for a dataset
  def gradient(self, dataSet):
    gradient = []
    costZero = self.cost(dataSet)
    for layer in range(0,len(self.weights)):
      for node in range(0,len(self.weights[layer])):
        for weight in range(0,len(self.weights[layer][node])):
          weightZero = self.weights[layer][node][weight]
          self.setWeight([layer,node,weight],weightZero+self.accuracy)
          gradient.append((self.cost(dataSet)-costZero)/self.accuracy)
          self.setWeight([layer,node,weight],weightZero)
    return gradient

  #optimize function for a dataset
  def optimize(self, dataSet):
    g = self.gradient(dataSet)
    i = 0
    for layer in range(0,len(self.weights)):
      for node in range(0,len(self.weights[layer])):
        for weight in range(0,len(self.weights[layer][node])):
          self.setWeight([layer,node,weight],self.weights[layer][node][weight]-self.learningRate*g[i])
          i += 1
    return self.cost(dataSet)
