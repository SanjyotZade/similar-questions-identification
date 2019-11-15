# Copyright (C) 13/11/19 sanjyotzade

#imports
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"


class model_architectures():

    def __init__(self):
        # model training constants
        self.LSTM1_EMBEDDING_DIM = 128
        self.DROPOUT_FRACTION = 0.2
        self.EPOCHS = 30
        self.BATCH_SIZE = 2048

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

        self.WORDS_IN_EMBEDDING = min(self.N_VOCAB, len(self.tokenizer.word_index))

    def seq_model(self, embedding_matrix):

        # Input Layers
        question1 = layers.Input(shape=(self.MAX_QUESTION_LENGTH,), dtype='int32', name="question1")
        question2 = layers.Input(shape=(self.MAX_QUESTION_LENGTH,), dtype="int32", name="question2")

        # Hidden Layers
        question1_embedding = layers.Embedding(input_dim=self.WORDS_IN_EMBEDDING + 1,
                                               output_dim=self.EMBEDDING_DIM,
                                               name="words_embedding_que1")(question1)

        question1_lstm0 = layers.Bidirectional(layers.LSTM(self.LSTM1_EMBEDDING_DIM ,
                                                           dropout=self.DROPOUT_FRACTION,
                                                           recurrent_dropout=self.DROPOUT_FRACTION,
                                                           return_sequences=True,
                                                           name="LSTM0_que1"), merge_mode="sum")(question1_embedding)

        question1_lstm1 = layers.Bidirectional(layers.LSTM(self.LSTM1_EMBEDDING_DIM ,
                                                           dropout=self.DROPOUT_FRACTION,
                                                           recurrent_dropout=self.DROPOUT_FRACTION,
                                                           return_sequences=True,
                                                           name="LSTM1_que1"), merge_mode="sum")(question1_lstm0)

        question2_embedding = layers.Embedding(input_dim=self.WORDS_IN_EMBEDDING + 1,
                                               output_dim=self.EMBEDDING_DIM,
                                               name="words_embedding_que2")(question2)

        question2_lstm0 = layers.Bidirectional(layers.LSTM(self.LSTM1_EMBEDDING_DIM ,
                                                           dropout=self.DROPOUT_FRACTION,
                                                           recurrent_dropout=self.DROPOUT_FRACTION,
                                                           return_sequences=True,
                                                           name="LSTM0_que2"), merge_mode="sum")(question2_embedding)

        question2_lstm1 = layers.Bidirectional(layers.LSTM(self.LSTM1_EMBEDDING_DIM ,
                                                           dropout=self.DROPOUT_FRACTION,
                                                           recurrent_dropout=self.DROPOUT_FRACTION,
                                                           return_sequences=True,
                                                           name="LSTM1_que2"), merge_mode="sum")(question2_lstm0)
        attention = layers.dot([question1_lstm1, question2_lstm1], [1, 1])
        attention = layers.Flatten()(attention)
        attention = layers.Dense((self.LSTM1_EMBEDDING_DIM * self.MAX_QUESTION_LENGTH))(attention)
        attention = layers.Reshape((self.MAX_QUESTION_LENGTH, self.LSTM1_EMBEDDING_DIM))(attention)

        merged = layers.add([question1_lstm1, attention])
        merged = layers.Flatten()(merged)
        merged = layers.Dense(200, activation='relu')(merged)
        merged = layers.Dropout(self.DROPOUT_FRACTION)(merged)
        merged = layers.BatchNormalization()(merged)
        merged = layers.Dense(200, activation='relu')(merged)
        merged = layers.Dropout(self.DROPOUT_FRACTION)(merged)
        merged = layers.BatchNormalization()(merged)
        merged = layers.Dense(200, activation='relu')(merged)
        merged = layers.Dropout(self.DROPOUT_FRACTION)(merged)
        merged = layers.BatchNormalization()(merged)
        merged = layers.Dense(200, activation='relu')(merged)
        merged = layers.Dropout(self.DROPOUT_FRACTION)(merged)
        merged = layers.BatchNormalization()(merged)

        # Output Layer
        predict = layers.Dense(1, activation="sigmoid", name="final_layer")(merged)
        model = Model(inputs=[question1, question2], outputs=predict)
        for layer in model.layers:
            if layer.name in ["words_embedding_que1", "words_embedding_que2"]:
                layer.set_weights([embedding_matrix])
                layer.trainable = False

        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
        return model

    # function to plot model training logs
    def plot_training_history(self, history_dict, plot_val=True, accuracy=True, chart_type="--o", save_path=False):
        if accuracy:
            acc = history_dict['acc']
        loss = history_dict['loss']

        if plot_val:
            if accuracy:
                val_acc = history_dict['val_acc']
            val_loss = history_dict['val_loss']
        epochs = range(1, len(loss) + 1)

        # visualize model training
        if accuracy:
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            axs[0].plot(epochs, loss, chart_type, label='Training loss')
            if plot_val:
                axs[0].plot(epochs, val_loss, chart_type, label='Validation loss')
                axs[0].set_title('training & validation loss')
            else:
                axs[0].set_title('training loss')
            axs[0].legend()
        else:
            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            axs.plot(epochs, loss, chart_type, label='Training loss')
            if plot_val:
                axs.plot(epochs, val_loss, chart_type, label='Validation loss')
                axs.set_title('training & validation loss')
            else:
                axs.set_title('training loss')
            axs.legend()

        if accuracy:
            axs[1].plot(epochs, acc, chart_type, label='Training acc')
            if plot_val:
                axs[1].plot(epochs, val_acc, chart_type, label='Validation acc')
                axs[1].set_title('training & validation accuracy')
            else:
                axs[1].set_title('training accuracy')
            axs[1].legend()

        if save_path:
            plt.savefig(save_path)
        plt.show()
        plt.close()
        return


if __name__ == "__main__":

    model_trainig_iteration_name = "glove_seq2"

    model_creation_obj = model_architectures()

    # create model iteration folder
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir_path = os.path.join(dir_path, model_trainig_iteration_name)
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    # trainning data
    train_question1 = np.load(open(os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.QUE1_TRAIN_DATA_FILE), 'rb'))
    train_question2 = np.load(open(os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.QUE2_TRAIN_DATA_FILE), 'rb'))
    train_targets = np.load(open(os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.TARGETS_TRAIN_DATA_FILE), 'rb'))

    test_question1 = np.load(open(os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.QUE1_TEST_DATA_FILE), 'rb'))
    test_question2 = np.load(open(os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.QUE2_TEST_DATA_FILE), 'rb'))
    test_targets = np.load(open(os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.TARGETS_TEST_DATA_FILE), 'rb'))

    embedding_weights_path = os.path.join(model_creation_obj.DIR_PATH, model_creation_obj.WORD_EMBEDDING_MATRIX_FILE)
    embedding_weights = np.load(open(embedding_weights_path, 'rb'))

    model = model_creation_obj.seq_model(embedding_weights)

    # This callback will save the current weights after every epoch
    # The name of weight file contains epoch number, val accuracy
    file_path = model_dir_path+"/weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoints = ModelCheckpoint(
        filepath=file_path,  # Path to the destination model file
        monitor='val_loss',
        save_best_only=False,
    )

    model.summary()

    history = model.fit([train_question1, train_question2],
                        train_targets,
                        epochs=model_creation_obj.EPOCHS,
                        validation_data=([test_question1, test_question2], test_targets),
                        batch_size=model_creation_obj.BATCH_SIZE, callbacks=[checkpoints])
    model_creation_obj.plot_training_history(history.history, save_path=os.path.join(model_dir_path, "training_stat.png"))
