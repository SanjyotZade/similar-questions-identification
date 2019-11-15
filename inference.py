# Copyright (C) 13/11/19 sanjyotzade

from data_processing import data_preparation
from tensorflow.keras.models import load_model


class prediction:

    def __init__(self, path_to_model):
        self.model = path_to_model
        self.seq_model = load_model(path_to_model)

    def que_similarity(self, question1, question2):
        data_prep_obj = data_preparation()

        question1 = data_prep_obj.tokenize_and_pad([question1])
        question2 = data_prep_obj.tokenize_and_pad([question2])
        pred = self. seq_model.predict([question1, question2])

        if pred[0] >= 0.5:
            return 1
        else:
            return 0


if __name__ == "__main__":

    que1 = "Why nobody answer my questions in Quora?"
    que2 = "Why is no one answering my questions in Quora?"
    label = 1
    path_to_model = "./glove_2seq/quora_weights.h5"
    pred_obj = prediction(path_to_model)
    print (pred_obj.que_similarity(que1, que2))
