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
    