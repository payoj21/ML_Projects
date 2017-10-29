import numpy
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
    def __init__(self, train_x, train_y, test_x, test_y, dict_size=5000, example_length=300, embedding_length=32, epoches=5, batch_size=128):
        '''
        initialize RNN model
        :param train_x: training data
        :param train_y: training label
        :param test_x: test data
        :param test_y: test label
        :param epoches:
        :param batch_size:
        '''
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
        self.model.add(Embedding(self.dict_size,output_dim=self.embedding_len))
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
    print('Accuracy for LSTM: ',accuracy)