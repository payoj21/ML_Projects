import argparse
import random
from collections import namedtuple
import numpy as np
from knn import Knearest, Numbers
import matplotlib.pyplot as plot


random.seed(20170830)


# READ THIS FIRST
# In n-fold cross validation, all the instances are split into n folds
# of equal sizes. We are going to run train/test n times.
# Each time, we use one fold as the testing test and train a classifier
# from the remaining n-1 folds.
# In this homework, we are going to split based on the indices
# for each instance.

# SplitIndices stores the indices for each train/test run,
# Indices for the training set and the testing set 
# are respectively in two lists named 
# `train` and `test`.

SplitIndices = namedtuple("SplitIndices", ["train", "test"])

def split_cv(length, num_folds):
    """
    This function splits index [0, length - 1) into num_folds (train, test) tuples.
    """


    # Finish this function to populate `folds`.
    # All the indices are split into num_folds folds.
    # Each fold is the testing set in a split, and the remaining indices
    # are added to the corresponding training set.

    splits = [SplitIndices([], []) for _ in range(num_folds)]
    indices = list(range(length))
    random.shuffle(indices)

    indices_division = [[] for x in range(num_folds)]       # dividing the indices elements into n_folds lists
    if(length % num_folds == 0):
        indices_division_set_size = length / num_folds          # size of each list
    else:
        indices_division_set_size = length / num_folds + 1

    for index in range(len(indices)):
        indices_division[int(index / indices_division_set_size)].append(indices[index])

    count_test = 0
    for split in splits:
        for i in indices_division[count_test]:              # taking 1 list from indices_division for test and rest lists for training
            split.test.append(i)
        for count in range(len(indices_division)):
            if count != count_test:
                for i in indices_division[count]:
                    split.train.append(i)
        count_test = count_test + 1

    return splits


def cv_performance(x, y, num_folds, k):
    """This function evaluates average accuracy in cross validation."""
    length = len(y)
    splits = split_cv(length, num_folds)
    accuracy_array = []
    accuracy = 0
    for split in splits:
        # Finish this function to use the training instances 
        # indexed by `split.train` to train the classifier,
        # and then store the accuracy 
        # on the testing instances indexed by `split.test`

        train_data_indices = split.train
        test_data_indices = split.test
        data_train = []
        data_test = []
        output_train = []
        output_test = []
        for i in train_data_indices:
            data_train.append(x[i])
            output_train.append(y[i])
        for i in test_data_indices:
            data_test.append(x[i])
            output_test.append(y[i])
    #    classify = Knearest.classify(data_train)
        knn_train = Knearest(data_train,output_train,k)

    #    print(data_train)
        confusion = knn_train.confusion_matrix(data_test,output_test)
        accuracy = knn_train.accuracy(confusion)
        accuracy_array.append(accuracy)

    return np.mean(accuracy_array)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()
    
    data = Numbers("../data/mnist.pkl.gz")

#    limit_array = [500,1000,5000,10000,15000,20000,25000]
    accuracy_array = []

    #for limit in limit_array:
    x, y = data.train_x, data.train_y
    print("Working with {0} examples".format(args.limit))
    if args.limit > 0:
        x, y = x[:args.limit], y[:args.limit]
    best_k, best_accuracy = -1, 0
    for k in [1, 3, 5, 7, 9]:
        accuracy = cv_performance(x, y, 5, k)
        print("%d-nearest neighbor accuracy: %f" % (k, accuracy))
        if accuracy > best_accuracy:
            best_accuracy, best_k = accuracy, k
    knn = Knearest(x, y, best_k)
    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    accuracy = knn.accuracy(confusion)
    #accuracy_array.append(accuracy)
    print("Accuracy for chosen best k= %d: %f\n" % (best_k, accuracy))

    #print(accuracy_array)

    #plot.plot(limit_array,accuracy_array,"Accuracy Plot")
    #plot.ylabel()
    #plot.suptitle("Limit vs Accuracy Plot")
    #plot.xlabel("Limits")
    #plot.ylabel("Accuracy")
    #plot.show()