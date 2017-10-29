from knn import Knearest, Numbers
import matplotlib.pyplot as plt


K_neighbor = [1,3,5,7,9,11,13,15,17,19,21]
accuracy_array = []
limit = 10000
data = Numbers("../data/mnist.pkl.gz")

for k in K_neighbor:
    if limit > 0:
        print("K-Neighbor: %i" % k)
        knn = Knearest(data.train_x[:limit], data.train_y[:limit],
                       k)
    else:
        knn = Knearest(data.train_x, data.train_y, k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.train_x[:limit], data.train_y[:limit])

    accuracy = knn.accuracy(confusion)
    print("Accuracy: %f" % accuracy)
    accuracy_array.append(accuracy*100)

print(accuracy_array)
plt.plot(K_neighbor, accuracy_array, "ro")
plt.suptitle("K-Neighbors vs Accuracy Plot")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.show()