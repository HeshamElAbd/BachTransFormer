#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:00:22 2019

@author: Hesham EL Abd
@Description: Prepearing the Midi file into traning ready 
Tensors.
@Note: 
    1-Data preprocessing is inspired by the post of 
    Sigurður Skúli in his artical@ 
    https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
    2- Data was download from https://www.reddit.com/r/WeAreTheMusicMakers/comments/3ajwe4/the_largest_midi_collection_on_the_internet/
"""
# import the models:
import music21
import tensorflow as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
# reading the data
notes=[]
midiFiles=["data/"+midiFile for midiFile in os.listdir("data/") if ".mid" in midiFile ]
numberOfFiles=len(midiFiles)
print("Number of Files to parse: {}".format(numberOfFiles))
counter=0
for midiFile in midiFiles:
    try:
        midiStream = music21.converter.parse(midiFile).flat.notes
        for element in midiStream:
            if isinstance(element, music21.note.Note):
                notes.append(str(element.pitch)+"$"+str(element.duration.quarterLength))
            elif isinstance(element, music21.chord.Chord):
                notes.append('-'.join(str(n) for n in element.normalForm)
                +"$"+str(element.duration.quarterLength))
        notes.append("%")
        counter+=1
        print("Number of processed files is {} out of {}".format(
             counter, numberOfFiles
           ))
    except:
        print("I had a problem with file {}".format(midiFile))
        pass
print("The number of elements in the corpa is {}".format(len(notes)))
print("Number Of unique elment in the corpa is {}".format(len(set(notes))))
# save a local copy of the corpa to the hardDisk for later usage:
with open("MidiDataSetAsList.pickle","wb") as output_:
    pickle.dump(notes,output_)











