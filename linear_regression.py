import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 random points in [0, 2]
y = 4 + 3 * X + np.random.randn(100, 1)  # Linear relation with noise

# Step 2: Prepare the DataFrame
data = pd.DataFrame(np.hstack((X, y)), columns=['Feature', 'Target'])

# Step 3: Implement the Linear Regression Model
class LinearRegression:
    def __init__(self):
        self.bias = 0
        self.weight = 0

    def fit(self, X, y, learning_rate=0.01, n_iterations=1000):
        n_samples = X.shape[0]
        for _ in range(n_iterations):
            y_predicted = self.weight * X + self.bias
            # Compute gradients
            weight_grad = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            bias_grad = (1/n_samples) * np.sum(y_predicted - y)
            # Update parameters
            self.weight -= learning_rate * weight_grad
            self.bias -= learning_rate * bias_grad

    def predict(self, X):
        return self.weight * X + self.bias

# Step 4: Train the model
model = LinearRegression()
model.fit(X, y)

# Step 5: Make predictions
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# Step 6: Plot the results
plt.scatter(data['Feature'], data['Target'], color='blue', label='Data points')
plt.plot(X_new, y_predict, color='red', label='Prediction')
plt.title('Linear Regression Example')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.legend()
plt.show()