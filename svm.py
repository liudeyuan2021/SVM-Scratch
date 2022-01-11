import numpy as np
import sys
import time

# X is the data matrix (each row is a data point)
# Y is desired output (1 or -1)


class myPegasos ():
    """ class myPegasos which implements SVM using 
        Stochastic Gradient Descent implemneted with 
        the Pegasos algorithm for training linear SVM.
    Attributes:
        lossf (list): store primal objective function value for each iteration
        w (numpy.array): trained weight
        y (numpy.array): class labels
    """

    def __init__(self, X, y, _lambda, k):
        '''initialize classifer and train model
        Args:
            X(numpy.array): Data Matrix with mxn where m is the
            y (numpy.array): Target vector (1,-1)
            _lambda = regularization paramter
            k: training sample size for each iteration
        '''

        self.lossf = []
        self.w = self.optimize(X, y, _lambda, k)

    def optimize(self, X, y, _lambda, k):
        """estimate the weight vector and intercept given X, y, _lambda, k
            Args:
                X (numpy.array): Data Matrix
                y (numpy.array): Response Vector
            Returns:
                numpy.array: trained weight vector w
        """

        i_weight = np.zeros(X.shape[1])
        i_weight.fill(np.sqrt(1 / _lambda / X.shape[1]))
        rand_neg = np.random.randint(
            0, high=X.shape[1], size=int(X.shape[1] / 2))
        i_weight[rand_neg] = -i_weight[rand_neg]

        w_current = i_weight

        ite = 0
        for t in range(1, 100 * X.shape[0]):
            ite = ite + 1
            if k != 1:
                X_t, y_t = self.selectk_data(X, y, k)

            else:
                ind = np.random.randint(0, high=X.shape[0], size=k)
                X_t, y_t = X[ind], y[ind]
            inds = np.dot(X_t, w_current) * y_t < 1

            X_sub = X_t[inds, :]
            y_sub = y_t[inds]

            const_t = 1.0 / _lambda / (t)
            weight = (1 - const_t * _lambda) * \
                w_current + const_t / k * np.dot(y_sub, X_sub)
            mini = np.array([1, 1 / np.sqrt(_lambda) /
                             np.sqrt(np.sum(weight**2))])

            w_new = np.amin(mini) * weight

            self.lossf.append(self.lossfxns(X, y, w_new, _lambda))

            if sum((w_new - w_current) ** 2) < 0.01:

                break
            else:
                w_current = w_new

        print(ite, ' iterations')
        return w_current

    def selectk_data(self, X, y, k):
        """Select training set for each iteration
        Args:
            X (numpy.array): Data Matrix
            y (numpy.array): Response Vector
            k (int): training sample size for each iteration
        Returns:
            (numpy.array, numpy.array): (X_k, y_k)
        """
        percent = k / 2000.0
        X_1 = X[y == 1]
        X_2 = X[y == -1]
        y_1 = y[y == 1]
        y_2 = y[y == -1]

        inds1 = np.random.randint(
            0, high=X_1.shape[0], size=round(percent * X_1.shape[0]))
        inds2 = np.random.randint(
            0, high=X_2.shape[0], size=round(percent * X_2.shape[0]))

        X_k = np.concatenate((X_1[inds1, :], X_2[inds2, :]))
        y_k = np.concatenate((y_1[inds1], y_2[inds2]))
        return X_k, y_k

    def predict(self, X, y):
        """Predict the class label for each observation in
        the data X and calculate error based on response vector y
        Args:
            predicted(numpy.array, numpy.array):
        Returns:
            float: error based on predicted class vs response vector
        """
        prediction = np.dot(self.w[:, np.newaxis].T, X.T)
        prediction = np.squeeze(prediction)

        predict = np.zeros_like(y)
        for i in range(prediction.shape[0]):
            if prediction[i] > 0:
                predict[i] = 1
            else:
                predict[i] = -1

        errors = np.array([predict[i] != y[i] for i in range(0, y.shape[0])])
        totalerror = np.sum(errors) / y.shape[0]
        return totalerror

    def lossfxns(self, X, y, w, _lambda):
        """Calcualte the loss function for the SVM problem
            Args:
                X (numpy.array): Design Matrix (m x n)
                y (numpy.array): Response Vector (1 x m)
                weight (numpy.array): weight parameter (1 x n)
                _lambda (float): regularization parameter
            Returns:
                float: current value of the primal objective function
        """
        tmp = 1 - y * np.dot(X, w)
        lossf = sum(tmp[(1 - y * np.dot(X, w)) > 0]) / X.shape[0] + \
            _lambda / 2 * np.dot(w, w)
        return lossf


def featureNormalize(X):
    """Preprocesses the data by subtracing the mean and dividing over std.axis
        Also adds intercept into feature
        Args:
            X (numpy.array): Data Matrix (m x n)
        Returns:
            numpy.array: Processed Data Matrix
    """
    stds = X.std(axis=0)
    newX = np.delete(X, np.where(stds == 0), 1)

    stds = newX.std(axis=0)
    means = newX.mean(axis=0)
    newX = (newX - means) / stds

    return np.concatenate((np.ones((newX.shape[0], 1)), newX), axis=1)


if __name__ == "__main__":
    X, y = read_file(sys.argv[1])
    X = featureNormalize(X)

    times = []
    numruns = int(sys.argv[3])
    k = int(sys.argv[2])

    for i in range(numruns):
        print("Training model ", i + 1, ' out of ',
              numruns, "...")

        begin = time.time()
        mP = myPegasos(X, y, 1e-4, k)

        end = time.time()
        times.append(end - begin)

        print('Runtime: ', round(end - begin, 3), ' seconds')

        print("\n")
    # Print combined error rates for each train set
    # percent averaged by the number of folds that ran
    print("------FINAL RESULT -------")
    print('Average runtime w/ minibatch size of ', k,
          ':\t', round(np.mean(times), 3), " sec.")

    print('STD runtime w/ minibatch size of ', k,
          ':\t', round(np.std(times), 3), " sec.")