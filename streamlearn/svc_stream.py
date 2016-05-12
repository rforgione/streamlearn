import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics.pairwise import euclidean_distances

class SVCStream(SGDClassifier):
    """
    m^2 memory is required to hold an RBF kernel matrix of an m row by n column matrix,
    as opposed to m*n memory required for the original matrix.
    For matrices where m >> n, this can result in a drastic increase in memory consumption.
    This RBF kernel computation allows us to stream one row in the
    matrix at a time, which allows us to limit memory consumption to
    the original m by n matrix. This is accomplished by using stochastic
    gradient descent and partial fitting, allowing us to compute the rbf kernel
    value for a single row relative to the entire original dataset, and perform
    gradient descent on that row to update the parameters of the model. Using
    this method, only one row of length m is needed at a time, which saves
    (m - n)*m - m = m^2 - nm - m = m(m - n - 1) in memory. When
    the difference between m and n is very large, this can amount to a large
    memory savings.

    There is almost certainly a loss in efficiency in using this method, as we
    are unable to take full advantage of the highly optimized matrix algebra
    functions in numpy and scikit-learn to compute the entire gram matrix in
    one batch. Further research / analysis is required to determine the full
    implications of this loss in efficiency.
    """
    def __init__(self, kernel='rbf', gamma=None):
        self.kernel = kernel

        self.gamma = None
        if gamma:
            self.gamma = gamma

        super(SVCStream, self).__init__(loss="hinge")
        self.X = None

    def rbf_kernel_single_row(self, row):

        if type(row) == int:
            row = self.X[row]

        return np.exp(-self.gamma*euclidean_distances(X=self.X, Y=row, squared=True)).reshape(1,-1)

    def stream_fit(self, X, y):
        self.X = X

        if not self.gamma:
            self.gamma = 1.0 / self.X.shape[0]

        for i in range(self.X.shape[0]):
            self.partial_fit(self.rbf_kernel_single_row(row=i),np.array([y[i]]), [0,1])

    def stream_predict(self, newX):
        return map(lambda i: self.predict(self.rbf_kernel_single_row(row=i)), range(newX.shape[0]))

