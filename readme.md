# Similar question identification

# Problem statement
Quora Question Pairs
Can you identify question pairs that have the same intent?

# Description
Many people on Quora ask questions with same intent in different wordings. 
For better user experience on quora, these questions should be merged togather so that writers do not need to answer same question multiple times.
Also it will make discovery of content easier on quora. Your task is to create a deep learning model to identify duplicate questions.

# Strategy
Train a binary classifier with two seperate embedding layer each for question1 and question2. The embedding layers are innitialized with
glove 300d vectors. The model is trained with binary cross entropy, predicting if the questions are same or different.

## Data creation process
- splitting data into train test
- questions's data encoding, padding and trauncating
- creating embedding weights

## training the model on the created dataset

 ![model arc](glove_2seq/training_stat.png)

Training the model yields accuracy og ~82% on the test dataset.

## Further improvements:
- training a siamese network with triplet loss
- using bert fine tuning for similarity prediction

## code understanding

- data_processing.py: A script with methods for text data creation for model training.
- model.py: A script with model architecture and training code.
- inference.py: c two sentences as input and binary(yes/no) output indicating if sentences are duplicates



