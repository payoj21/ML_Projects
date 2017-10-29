import argparse
import numpy as np
import pickle, gzip
import matplotlib.pyplot as plt

class Network:
    def __init__(self, sizes):
        self.L = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(n, 1) for n in self.sizes[1:]]
        self.weights = [np.random.randn(n, m) for (m,n) in zip(self.sizes[:-1], self.sizes[1:])]

    def g(self, z):
        """
        activation function
        """
        return sigmoid(z)

    def g_prime(self, z):
        """
        derivative of activation function
        """
        return sigmoid_prime(z)

    def forward_prop(self, a):
        """
        memory aware forward propagation for testing
        only.  back_prop implements it's own forward_prop
        """
        for (W,b) in zip(self.weights, self.biases):
            a = self.g(np.dot(W, a) + b)
        return a

    def gradC(self, a, y):
        """
        gradient of cost function
        Assumes C(a,y) = (a-y)^2/2
        """
        return (a - y)

    def SGD_train(self, train, epochs, eta, lam=0.0, verbose=True, test=None):
        """
        SGD for training parameters
        epochs is the number of epocs to run
        eta is the learning rate
        lam is the regularization parameter
        If verbose is set will print progressive accuracy updates
        If test set is provided, routine will print accuracy on test set as learning evolves
        """
        n_train = len(train)
        epoch_accuracy_list = [] #omit
        hidden_acc_train = 0    #omit
        hidden_acc_test = 0     #omit
        for epoch in range(epochs):
            perm = np.random.permutation(n_train)
            for kk in range(n_train):
                xk = train[perm[kk]][0]
                yk = train[perm[kk]][1]
                dWs, dbs = self.back_prop(xk, yk)
                # TODO: Add L2-regularization

                self.weights = [W - eta*(dW + lam*W) for (W, dW) in zip(self.weights, dWs)] # L2 regularization
                                                                                            # (subtracting eta*lam*W)
                self.biases = [b - eta*(db) for (b, db) in zip(self.biases, dbs)]   # No need to regularize Bias.
                                                                                            # Overfitting doesn't occur due to biases
            if verbose:
                if epoch==0 or (epoch + 1) % 15 == 0:
                    acc_train = self.evaluate(train)
                    if test is not None:
                        acc_test = self.evaluate(test)
                        #epoch_accuracy_list.append((acc_test,acc_train))    #omit
                        print("Epoch {:4d}: Train {:10.5f}, Test {:10.5f}".format(epoch+1, acc_train, acc_test))
                    else:
                        print("Epoch {:4d}: Train {:10.5f}".format(epoch+1, acc_train))
            # if epoch == epochs-1:                   #omit the whole if condition
            #     hidden_acc_train = self.evaluate(train)
            #     hidden_acc_test = self.evaluate(test)

        #return epoch_accuracy_list, (hidden_acc_train,hidden_acc_test)  #omit this return
    def back_prop(self, x, y):
        """
        Back propagation for derivatives of C wrt parameters
        """
        db_list = [np.zeros(b.shape) for b in self.biases]
        dW_list = [np.zeros(W.shape) for W in self.weights]

        a = x
        a_list = [a]
        z_list = [np.zeros(a.shape)] # Pad with throwaway so indices match

        for W, b in zip(self.weights, self.biases):
            z = np.dot(W, a) + b
            z_list.append(z)
            a = self.g(z)
            a_list.append(a)

        # Back propagate deltas to compute derivatives
        # TODO delta  =
        delta = (a_list[self.L-1] - y)*self.g_prime(z_list[self.L-1]) # calculating delta (Error) for the output layer which is
                                                                      # delta(L) = (a-y)*(a)*(1-a)

        for ell in range(self.L-2,-1,-1):
            # TODO db_list[ell] =
            db_list[ell] = delta        #change in bias is just delta
            # TODO dW_list[ell] =
            dW_list[ell] = np.dot(delta,a_list[ell].T)      #change in weights is delta.transpose of a(l)
            # TODO delta =
            delta= np.dot(self.weights[ell].T,delta)*self.g_prime(z_list[ell])  #delta for this layer =
                                                                                        # weight(l)_transpose.delta(l+1)*g'(z(l))
            pass

        return (dW_list, db_list)

    def evaluate(self, test):
        """
        Evaluate current model on labeled test data
        """
        ctr = 0
        for x, y in test:
            yhat = self.forward_prop(x)
            ctr += np.argmax(yhat) == np.argmax(y)
        return float(ctr) / float(len(test))

    def compute_cost(self, x, y):
        """
        Evaluate the cost function for a specified
        training example.
        """
        a = self.forward_prop(x)
        return 0.5*np.linalg.norm(a-y)**2

def sigmoid(z, threshold=20):
    z = np.clip(z, -threshold, threshold)
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def mnist_digit_show(flatimage, outname=None):

    import matplotlib.pyplot as plt

    image = np.reshape(flatimage, (-1,14))

    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if outname:
        plt.savefig(outname)
    else:
        plt.show()

if __name__ == "__main__":

    f = gzip.open('../data/tinyMNIST.pkl.gz', 'rb') # change path to ../data/tinyMNIST.pkl.gz after debugging
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    train, test = u.load()
    input_dimensions = len(train[0][0])
    output_dimensions = len(train[0][1])
    #hidden_layer_dimensions = [10,25,50,75,100,125,150,200]
    hidden_layer_dimensions = 50                                          #omit
    print('Number of Input Features: ',input_dimensions)
    #print('Hidden Layer Dimensions: ', hidden_layer_dimensions)
    print('Number of Output classes: ', output_dimensions)
    # hidden_layer_accuracy = []                                              #omit
    # number_of_epochs = [1,15,30,45,60,75,90,105,120,135,150,165,180,195]    #omit
    #for hd in hidden_layer_dimensions:                                      #omit
    print('\nHidden Layer Dimensions: ',hidden_layer_dimensions)                             #omit
    nn = Network([input_dimensions,hidden_layer_dimensions,output_dimensions])
    nn.SGD_train(train, epochs=200, eta=0.1, lam=0.0001, verbose=True, test=test) #edit this
    #print(hidden_accuracy)                                              #omit
    #hidden_layer_accuracy.append(hidden_accuracy)                       #omit

    # plt.plot(number_of_epochs, epoch_accuracy, 'ro')
    # plt.suptitle("Number of Epochs vs Train and Test Accuracy Plot")
    # plt.xlabel("Number of Epochs")
    # plt.ylabel("Accuracy")
    # plt.show()

    # plt.plot(hidden_layer_dimensions,hidden_layer_accuracy, 'ro', 'bo')
    # plt.suptitle("Hidden Layers dimensions vs Train and Test Accuracy Plot")
    # plt.xlabel("Hidden Layer Dimension")
    # plt.ylabel("Accuracy")
    # plt.show()
