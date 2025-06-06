import numpy as np
import matplotlib.pyplot as plt

def weight(x, xi, tau):
    return np.exp(-((x - xi)**2) / (2 * tau**2))

def predict(x, X, y, tau):
    w = [weight(x, xi, tau) for xi in X]
    W = np.diag(w)
    X_ = np.c_[np.ones(len(X)), X]
    x_ = np.array([1, x])
    theta = np.linalg.pinv(X_.T @ W @ X_) @ X_.T @ W @ y
    return x_ @ theta

X = np.linspace(0, 6, 50)
y = np.sin(X) + 0.1 * np.random.randn(50)
X_test = np.linspace(0, 6, 100)
y_pred = [predict(x, X, y, 0.3) for x in X_test]

plt.scatter(X, y)
plt.plot(X_test, y_pred)
plt.show()
