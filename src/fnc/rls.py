import numpy as np

class Estimator(object):
	# recursive least squares

	def __init__(self, theta0, M0):
		self.theta = theta0
		self.M = M0

	def update(self, x, y):
		if x.ndim < 2:
			x = x[:,np.newaxis]
		if y.ndim < 2:
			y = y[:,np.newaxis]
		K = np.linalg.pinv(self.M).dot(x) / (1 + x.T.dot(self.M).dot(x))
		self.theta = self.theta + K.dot(y.T - x.T.dot(self.theta))
		self.M = self.M + x.dot(x.T)