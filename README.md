# Machine Learning Final Project (Autumn 2019)

This repository contains the Final Project for Advanced Machine Learning 
for the Autumn semester of 2019 by Rafał Markiewicz (rafm@itu.dk), 
Sabin Rimniceanu (srim@itu.dk) and Jorgel Këci (joke@itu.dk)

## Implementation of CNN for Sentence Classification


Sentiment Analysis is a process of text classification, which analyzes whether a text has a positive, 
or a negative connotation.

This Project is an implementation of a CNN capable of performing a Sentiment Analysis on a dataset of Movie Reviews

## Prerequisites

* The project has been built and tested using Python 3.6. Please download it [here](https://www.python.org/downloads/).
* If running from the command line or Python Console, the following `pip` packages are required:
    * tensorflow (version 1.14)
    * numpy
* If the word2vec process were to be run, a gensim package would also be needed. 
However, since it is not used for anything but data extraction from word2vec files 
it is not needed to run this project

## Running the project

The simplest way to run the program is by navigating to the project root and typing `python runner.py`

Please make sure to run Python 3, thus if both 2 and 3 are installed the command may look like `python3 runner.py`

Also, please disregard the deprecation warnings coming from usage of tensorflow 1.14, 
since usage of Keras was prohibited for this project (which is strongly incorporated in tensorflow 2) 
we decided to use the outdated, yet functional tensorflow 1.14

## Data

Project data is structured as such:  
|-`data`  
|--`mr` - contains positive and negative data file for the MR dataset

|-`vectors`  
|--`processed`   
|---`mr`   
|----`mapping.csv` - contains mapping of all the words in mr dataset to indexes of their vectors   
|----`vectors.npy` - contains all of the word2vec vectors for MR dataset, 
plus randomly initialized vectors for words not found in word2vec

## Hyperparameters
If you wish to change any of the Hyperparameters of the project, they are all placed in the config.py file

