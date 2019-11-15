# Copyright (C) 13/11/19 sanjyotzade
import os
import pickle
import numpy as np
import pandas as pd


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class data_preparation():

    def __init__(self):
        # data prep constants
        self.DIR_PATH = os.path.realpath(os.path.dirname(__file__))
        self.DATA_PATH = os.path.join(self.DIR_PATH, "data")

        self.COMPLETE_DATA_NAME = "complete_que_pairs.csv"
        self.TRAIN_DATA_CSV = "ques_train.csv"
        self.TEST_DATA_CSV = "ques_test.csv"

        self.QUE1_TRAIN_DATA_FILE = 'q1_train.npy'
        self.QUE2_TRAIN_DATA_FILE = 'q2_train.npy'
        self.TARGETS_TRAIN_DATA_FILE = 'label_train.npy'

        self.QUE1_TEST_DATA_FILE = 'q1_test.npy'
        self.QUE2_TEST_DATA_FILE = 'q2_test.npy'
        self.TARGETS_TEST_DATA_FILE = 'label_test.npy'

        self.WORD_EMBEDDING_MATRIX_FILE = 'word_embedding_matrix.npy'
        self.NB_WORDS_DATA_FILE = 'nb_words.json'
        self.N_VOCAB = 200000
        self.MAX_QUESTION_LENGTH = 25
        self.EMBEDDING_DIM = 300

        if  os.path.exists(os.path.join(self.DIR_PATH, 'tokenizer.pickle')):
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)

    def create_load_tokenizer(self, que_data):
        if not os.path.exists(os.path.join(self.DIR_PATH, 'tokenizer.pickle')):
            print ("\nCreating and saving tokenizer")
            question1 = list(que_data["question1"])
            question2 = list(que_data["question2"])

            tokenizer = Tokenizer(num_words=self.N_VOCAB)
            tokenizer.fit_on_texts(question1+question2)  # feeding the text data to tokenizer

            print("Words in index: {}\n" .format(len(tokenizer.word_index)))
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.tokenizer = tokenizer
        return

    def train_test_split(self, complete_data, target_column_name, split_percentage=0.8):
        complete_data = complete_data.dropna()
        complete_data = complete_data.reset_index(drop=True)

        if not (os.path.exists(os.path.join(self.DIR_PATH, "data", "ques_train.csv"))
                or os.path.exists(os.path.join(self.DIR_PATH, "data", "ques_train.csv"))) :

            print("\nShape of entire dataset: {}".format(complete_data.shape))
            print("target distribution in entire dataset \n{}".format(complete_data[target_column_name].value_counts() / complete_data.shape[0]))
            np.random.seed(8732687)
            msk = np.random.rand(len(complete_data)) < split_percentage

            train = complete_data[msk]
            test = complete_data[~msk]
            train.to_csv("ques_train.csv", index=False)
            test.to_csv("ques_test.csv", index=False)

            print("\nShape of train dataset: {}".format(train.shape))
            print("target distribution in train dataset \n{}".format(
                train[target_column_name].value_counts() / train.shape[0]))

            print("\nShape of test dataset: {}".format(test.shape))
            print("target distribution in test dataset \n{}".format(
                test[target_column_name].value_counts() / test.shape[0]))
        return complete_data

    def tokenize_and_pad(self, data):
        data_sequence = self.tokenizer.texts_to_sequences(data)
        return pad_sequences(data_sequence, maxlen=self.MAX_QUESTION_LENGTH)

    def do_preprocessing(self, data_to_process, data_type="Train"):
        print ("\nProcessing {}".format(data_type))
        question1 = self.tokenize_and_pad(list(data_to_process["question1"]))
        question2 = self.tokenize_and_pad(list(data_to_process["question2"]))
        targets = np.array(data_to_process["is_duplicate"], dtype=int)

        if data_type == "Train":
            np.save(open(self.QUE1_TRAIN_DATA_FILE, 'wb'), question1)
            np.save(open(self.QUE2_TRAIN_DATA_FILE, 'wb'), question2)
            np.save(open(self.TARGETS_TRAIN_DATA_FILE, 'wb'), targets)
        else:
            np.save(open(self.QUE1_TEST_DATA_FILE, 'wb'), question1)
            np.save(open(self.QUE2_TEST_DATA_FILE, 'wb'), question2)
            np.save(open(self.TARGETS_TEST_DATA_FILE, 'wb'), targets)
        return question1, question2, targets

    def create_embedding_matrix(self, path_to_glove_data):

        if not os.path.exists(os.path.join(self.DIR_PATH, self.WORD_EMBEDDING_MATRIX_FILE)):
            print ("\nCreating word embedding matrix...")
            embeddings_index = {}
            with open(path_to_glove_data, encoding='utf-8') as f:
                for line in f:
                    values = line.split(' ')
                    word = values[0]
                    embedding = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = embedding
            print('Word embeddings: %d' % len(embeddings_index))
            word_index = self.tokenizer.word_index
            nb_words = min(self.N_VOCAB, len(word_index))
            word_embedding_matrix = np.zeros((nb_words + 1, self.EMBEDDING_DIM))
            for word, i in word_index.items():
                if i > self.N_VOCAB:
                    continue
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    word_embedding_matrix[i] = embedding_vector
            np.save(open(self.WORD_EMBEDDING_MATRIX_FILE, 'wb'), word_embedding_matrix)
            print('Null word embeddings: %d\n' % np.sum(np.sum(word_embedding_matrix, axis=1) == 0))
        return


if __name__ == "__main__":
    data_prep_obj = data_preparation()

    #creating train and test data
    question_data = pd.read_csv(os.path.join(data_prep_obj.DATA_PATH, data_prep_obj.COMPLETE_DATA_NAME))
    complete_data = data_prep_obj.train_test_split(question_data, target_column_name="is_duplicate")

    # create and save data tokenizer
    data_prep_obj.create_load_tokenizer(complete_data)

    # pre-process data with created tokenizer
    train_data = pd.read_csv(os.path.join(data_prep_obj.DIR_PATH, data_prep_obj.TRAIN_DATA_CSV))
    test_data = pd.read_csv(os.path.join(data_prep_obj.DIR_PATH, data_prep_obj.TEST_DATA_CSV))

    train_question1, train_question2, train_targets = data_prep_obj.do_preprocessing(train_data)
    test_question1, test_question2, test_targets = data_prep_obj.do_preprocessing(test_data, data_type="Test")

    # create embedding matrix
    glove_data_path = "/Users/sanjyotzade/Documents/similar-questions-identification/data/glove.840B.300d.txt"
    data_prep_obj.create_embedding_matrix(glove_data_path)




