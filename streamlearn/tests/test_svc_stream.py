import unittest
from streamlearn.svc_stream import SVCStream
import numpy as np

class TestSVCStream(unittest.TestCase):

    data = np.array([[1,2],[3,4],[7,9]])
    response = np.array([0,1,0])
    clf = SVCStream()

    def test_single_row_rbf(self):
        self.clf.stream_fit(self.data, self.response)
        assert self.clf.rbf_kernel_single_row(0).shape[0] == 1
        assert self.clf.rbf_kernel_single_row(0).shape[1] == 3

    def test_stream_predict(self):
        self.clf.stream_fit(self.data, self.response)
        assert self.clf.stream_predict(np.array([[1,2]]))
        assert self.clf.coef_.shape[1] == 3
