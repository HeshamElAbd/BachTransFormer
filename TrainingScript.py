#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:19:49 2019
@author: Hesham El Abd
@Description: The scripts loads the processed midi corpa, numerically encode it
and train a Transformer encoder on it. 
"""
# import modules:
import tensorflow as tf
import numpy as np
import pickle
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



















