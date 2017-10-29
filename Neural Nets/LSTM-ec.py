import numpy
import keras
import os
import sys
import numpy as np
import codecs

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


class RNN:
    '''
    RNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, dict_size=5000, example_length=500, embedding_length=32, epoches=15, batch_size=128):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epoches:
        :param batch_size:
        '''
        BASE_DIR = ''
        GLOVE_DIR = BASE_DIR + 'glove.6B/'

        embeddings_index = {}
        f = codecs.open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()




        index_from = 3
        word_to_id = keras.datasets.imdb.get_word_index()
        word_to_id = {k: (v + index_from) for k, v in word_to_id.items()}
        word_to_id["<PAD>"] = 0
        word_to_id["<START>"] = 1
        word_to_id["<UNK>"] = 2

        id_to_word = {value: key for key, value in word_to_id.items()}

        print(id_to_word)
        num_words = dict_size

        embedding_matrix = np.zeros((num_words,embedding_length))

        for w, i in enumerate(id_to_word):
            embedding_vector = embeddings_index.get(w)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector



        self.batch_size = batch_size
        self.epoches = epoches
        self.example_len  = example_length
        self.dict_size = dict_size
        self.embedding_len = embedding_length

        # TODO:preprocess training data

        self.train_x = sequence.pad_sequences(train_x, maxlen=self.example_len)
        self.test_x = sequence.pad_sequences(test_x, maxlen=self.example_len)
        self.train_y = train_y
        self.test_y = test_y


        # TODO:build model
        self.model = Sequential()
        embedding_layer = Embedding(self.dict_size, self.embedding_len, weights=[embedding_matrix])
        self.model.add(embedding_layer)
        self.model.add(LSTM(64, dropout=0.5))


        self.model.add(Dense(1, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self):
        '''
        fit in data and train model
        :return:
        '''
        # TODO: fit in data to train your model
        self.model.fit(self.train_x,self.train_y, batch_size=self.batch_size, epochs=self.epoches, verbose =1)
    def evaluate(self):
        '''
        evaluate trained model
        :return:
        '''
        return self.model.evaluate(self.test_x, self.test_y)


if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=5000)
    rnn = RNN(train_x, train_y, test_x, test_y, epoches=3)
    rnn.train()
    accuracy = rnn.evaluate()
    print('Accuracy: ',accuracy)

