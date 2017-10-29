from knn import Knearest, Numbers
import matplotlib.pyplot as plt


limit_array = [500,1000,5000,10000,15000,20000,25000,30000,35000,40000,45000,50000]

accuracy_array = []
data = Numbers("../data/mnist.pkl.gz")

for limit in limit_array:
    if limit > 0:
        print("Data limit: %i" % limit)
        knn = Knearest(data.train_x[:limit], data.train_y[:limit],
                       3)
    else:
        knn = Knearest(data.train_x, data.train_y, k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)

    accuracy = knn.accuracy(confusion)
    print("Accuracy: %f" % accuracy)
    accuracy_array.append(accuracy*100)



print(accuracy_array)
plt.plot(limit_array, accuracy_array, "ro")
plt.suptitle("K-Neighbors vs Accuracy Plot")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()
