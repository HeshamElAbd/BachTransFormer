#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:19:49 2019
@author: Hesham El Abd
@Description: The scripts loads the processed midi corpus, numerically encode it
and train a Transformer encoder on it. 
"""
# import modules:
import tensorflow as tf
import numpy as np
import pickle
from Models import EncoderModels
# define some global parameters:
batchSize=256
lengthOfcondString=90
# load the dataset
with open("MidiDataSetAsList.pickle","rb") as input_:
    dataSet=pickle.load(input_)
# Numerical encode the dataset:
uniqueMusicalElements=set(dataSet)
music2int=dict()
integerIndex=1
for musicalElement in uniqueMusicalElements:
    music2int[musicalElement]=integerIndex
    integerIndex+=1
# a reverse mapping dict to map from musical notes to integers:
int2music=dict()
for integerIndex in music2int.keys():
    int2music[integerIndex]=music2int[integerIndex]
# save the dictionary for later usage with the models during deployment:
with open("music2Int.pickle","wb") as output_:
    pickle.dump(music2int,output_)
with open("int2music.pickle","wb") as output_:
    pickle.dump(int2music,output_)
# Encode bach corpa
EncodedMusic=np.array([music2int[musicElement] for musicElement in dataSet])
# prepear a dataset from the EncodedMusic
dataSet=tf.data.Dataset.from_tensor_slices(EncodedMusic).batch(
        batch_size=lengthOfcondString+1,drop_remainder=True)
def mapTrainingTensor(batch):
    """
    takes an encoded musical string of length lengthOfcondString+1 and
    return two strings of length lengthOfcondString, these two strings 
    represent the input and the target to the model  and are shift by one. 
    for example, if the input string is [145 67 789 234] the output is two
    string the first is 145 67 789 and the second 67 789 234. 
    """
    inputs=batch[:-1]
    output_=batch[1:]
    return inputs,output_
dataSet=dataSet.map(mapTrainingTensor).shuffle(1000).batch(batchSize)
## define the model
bachModeler=EncoderModels.Modeler(embedding_dim=64,
                            vocabulary_size=len(uniqueMusicalElements)+1,
                            conditional_string_length=lengthOfcondString,
                            num_encoder_layer=8,
                            num_heads=4,
                            num_neuron_pointwise=256,
                            rate=0.1,
                            return_attent_weights=True)
## define the loss and optimizer
lossFunc=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer=tf.keras.optimizers.Adam()
## define an accuracy metrics
accuracy=tf.keras.metrics.CategoricalAccuracy()
# define an input signature:
inputSignature=[
        tf.TensorSpec(shape=(None,lengthOfcondString),dtype=tf.int32),
        tf.TensorSpec(shape=(None,lengthOfcondString),dtype=tf.int32)]
## define training step function:
@tf.function(input_signature=inputSignature)
def trainStep(inputTensor,targetTensor):
    """
    Train the model on a batch of data and return its output
    """
    with tf.GradientTape() as tape:
        predictions=bachModeler(inputTensor,True)
        loss=lossFunc(y_true=targetTensor,y_pred=predictions)
    grads=tape.gradient(loss,bachModeler.trainable_variables)
    optimizer.apply_gradients(zip(grads,bachModeler.trainable_variables))
    accuracy(y_true=tf.one_hot(targetTensor,depth=len(uniqueMusicalElements)+1),
             y_pred=predictions)
    
inputSignatureTwo=[
        tf.TensorSpec(shape=(None,lengthOfcondString),dtype=tf.int32)]

@tf.function(input_signature=inputSignatureTwo)
def getAttentionWeights(inputTensor): 
    """
    The function return the self-attention weights of each layer in the Encoder
    part of the model for the given input Tensor
    ## inputs:
    # inputTensor: a 2D Tensor of shape (BatchSize,lengthOfcondString)
    ## outputs:
    # attentionWeights: a list of length num_encoder_layer where each element
    inside the list is a 4D tensor with 
    shape(batchSize,numOfHeads,seqLength,seqLength)
    """
    _, attentionWeights=bachModeler(inputTensor,False)
    for layerAttention in attentionWeights:
        assert len(layerAttention.shape)==4, """ Something went wrong during the
        calculations, the expected ranke of the tensor should be 4 however
        the current rank is {}""".format(len(layerAttention.shape))
    return attentionWeights

# storing a batch to mointer the evolution of self-attention weights
for batchToWatch,_ in dataSet.take(1): _


def trainEpoch(numOfEpoch,pathToSaveWeights):
    """
    The function train the model for a numOfEpoch and saves the weights of the
    model after each epoch at the pathToSaveWeights Path. After each epoch it
    calls the function getAttentionWeights with a fixed batch of data to model 
    how self-attention was evolving through training time.
    calls  
    ## inptus: 
    # numOfEpochs: is the number of epochs to train the model
    # pathToSaveWeights.
    ## ouputs:
    """























