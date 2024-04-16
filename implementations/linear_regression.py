import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

class LinearRegression:
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:

        self.X = X
        self.y = y.reshape((y.shape[0], 1))
        self.theta = np.zeros((X.shape[1], 1))

   
    def __cost_function(self, y_hat: np.ndarray) -> float:

        cost = 0.5 * np.sum(np.square(y_hat - self.y))

        return cost

    def __fit(self, epochs: int, lr: float) -> None:

        for epoch in range(epochs):

            #forward pass
            y_hat = np.dot(self.X, self.theta)            
            cost = self.__cost_function(y_hat=y_hat)
            print(f"Epoch {epoch}: Cost: {cost}")
            #backward pass with gradient descent
            self.__update_theta(y_hat=y_hat, lr=lr)
            
    def __update_theta(self, y_hat: np.ndarray, lr: float) -> None:

        #print((y_hat - self.y).shape, self.X.T.shape)                
        self.theta = self.theta - lr * (np.sum(np.dot((y_hat - self.y), self.X.T)) / self.X.shape[0])
       

    def fit(self, epochs: int, lr: float) -> None:

        self.__fit(epochs=epochs, lr=lr)

    def predict(self, X: np.ndarray) -> np.ndarray:

        return np.dot(X, self.theta)
        

#Simple test, just to check if the model fits the training data
X, y = make_regression(n_features=1, n_samples=50, noise=3, random_state=42)
lr = LinearRegression(X, y)

lr.fit(100, 0.01)

#print(f"Parameters: {lr.theta[0]}")

plt.scatter(X, y, color='red')
plt.title("Example Data")   
plt.xlabel("inputs")
plt.ylabel("targets")

plt.plot(X, lr.theta*X)

plt.savefig("linear_regression.png")