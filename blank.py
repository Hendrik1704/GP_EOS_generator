from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

# Example data
X_train = np.array([[1, 3], [2, 2], [3, 1]])
y_train = np.array([1, 2, 3])

# Gaussian Process with RBF kernel
kernel = RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=kernel)
gpr.fit(X_train, y_train)

# Test points
X_test = np.array([[1.5, 2.5], [2.5, 1.5]])

# Generate samples
samples = gpr.sample_y(X_test, n_samples=10, random_state=42)

print("Shape of samples:", samples.shape)
print("Samples:\n", samples)
