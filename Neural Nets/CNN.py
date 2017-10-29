
import argparse
import pickle
import gzip
import math
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import ZeroPadding2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import to_categorical
from keras.layers.core import Reshape


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches

        # TODO: reshape train_x and test_x
#        print('Length: ', train_y.shape[1])
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length

        # length of an input
        width_x_height = train_x.shape[1]

        width = int(math.sqrt(width_x_height))
        height = width

        #reshaping the train and test data so that it can be input to ConvNet
        self.train_x = train_x.reshape(train_x.shape[0], width, height, 1)
        self.test_x = test_x.reshape(test_x.shape[0], width, height, 1)

        # normalize data to range [0, 1] ALREADY NORMALIZED
        # self.train_x /= 255
        # self.test_x /= 255

        # TODO: one hot encoding for train_y and test_y
        # Since data is classified among 10 digits (0-9) therefore 10 classes
        self.train_y = to_categorical(train_y, 10)
        self.test_y = to_categorical(test_y, 10)


        # TODO: build you CNN model
        # 4 convolutional layer NN
        # here I have taken filter size as 5X5
        # accuracy changes with activation function
        # 2 Convolution Layer followed by MaxPooling. Tried AveragePooling: Accuracy dropped
        # Flatten is used to make the dimension of the output from Maxpooling acceptable for
        # Fully Connected layer

        self.model = Sequential()

        self.model.add(ZeroPadding2D((2, 2), input_shape=(width,height,1)))
        self.model.add(Conv2D(32, kernel_size=(5,5), strides= (1,1), activation='relu'))
        self.model.add(ZeroPadding2D((2, 2)))
        self.model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
        self.model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
        self.model.add(Dropout(0.5))
        #
        # self.model.add(ZeroPadding2D((2, 2)))
        # self.model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
        # self.model.add(ZeroPadding2D((2, 2)))
        # self.model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
        # self.model.add(MaxPool2D(pool_size=(2, 2), strides=(2,2)))
        # self.model.add(Dropout(0.5))



        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''
        # TODO: fit in training data
        self.model.fit(self.train_x, self.train_y, batch_size=self.batch_size, epochs=self.epoches, verbose = 1)
        pass

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y, verbose = 0)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

#    batch_size = [32,64,128,256,512]
#    accuracy = []
#    for size in batch_size:
    batch_size = 128
    epochs = 15
    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y, epoches=epochs, batch_size=batch_size)
    cnn.train()
    acc = cnn.evaluate()

    print('\nAccuracy: ',acc)
