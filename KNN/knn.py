
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import matplotlib.pyplot as plt


import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.


        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """
        
        # Finish this function to store necessary data so you can 
        # do classification later

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y value for
        # these indices
        #
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.median.html

        y_output = []
        for i in range(len(item_indices)):
            y_output.append(self._y[item_indices[i]])

        output = list(dict(Counter(y_output)).items())
        output.sort(key=lambda x: x[1], reverse = True)

        '''
        dictionary_of_y = dict()             # dict of {y:list of indices of _y}
        for i in range(len(item_indices)):   # populating the dictionary using item_indices and _y
            if self._y[item_indices[i]] not in dictionary_of_y:
                dictionary_of_y[self._y[item_indices[i]]] = [item_indices[i]]
            else:
                dictionary_of_y[self._y[item_indices[i]]].append(item_indices[i])

        list_output = list(dictionary_of_y.items())
        list_output.sort(key=lambda x: len(x[1]), reverse=True)  # sorting in descending order with parameter length of list of indices (longer the list of indices of a _y, more is the frequency)
    #    count = 1
        '''
        y_array = []    # array to find median, element of y_array has values of most frequent y(s)
        for i in range(0,len(output)):  # checking and adding if there are two or more ys with same frequency as the most one
            if(output[i][1] == output[0][1]):
    #            count = count+1
                y_array.append(output[i][0])


        #label_x = y_array[0]
        label_x = numpy.median(y_array)
        return int(label_x)

       # for i in range(len(item_indices)):
       #     item_indices[i] = list_output[0][1][i]

        # return self._y[item_indices[0]]

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        # Finish this function to find the k closest points, query the
        # majority function, and return the value.


    #    tree = BallTree(example, leaf_size=2)
        distance, index = self._kdtree.query([example], self._k)


        output_label =self.majority(index[0])

        return output_label
    #    return self.majority(list(random.randrange(len(self._y)) \
    #                              for x in range(self._k)))

    def confusion_matrix(self, test_x, test_y, debug=False):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrixfor the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            data_index += 1
            output_y = self.classify(xx)
            if output_y in d[yy]:
                d[yy][output_y] += 1
            else:
                d[yy][output_y] = 1
            if debug and data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total > 0:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code
#    limit_array = [500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
#    accuracy_array = []

#    for limit in limit_array:

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in range(10)))
    accuracy = knn.accuracy(confusion)
    print("Accuracy: %f" % accuracy)
#    accuracy_array.append(accuracy*100)
#    print(accuracy_array)
#    plt.plot(limit_array, accuracy_array, "ro")
#    plt.suptitle("Limit vs Accuracy Plot")
#    plt.xlabel("Limits")
#    plt.ylabel("Accuracy")
#    plt.show()