import numpy as np

#  Linear Regression code
class Linear:
    def __init__(self, A, b):
        self.A = A.copy()
        self.b = b.copy()

    def forward(self, x):
        self.x = x
        return np.dot(self.A, x) + self.b

    def backward(self, dy):
        dx = np.dot(self.A.T, dy)
        self.dA = np.outer(dy, self.x)
        self.db = dy.copy()
        return dx


#Task 2: Logistic Regression by Using Linear Regression
np.random.seed(0)

# Random inputs
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float)


A = np.random.randn(1, 2) * 0.01 #array of A
b = np.zeros(1) #array of b

linear = Linear(A, b)

lr = 0.1
losses = []

for epoch in range(50):
    total_loss = 0
    for i in range(len(X)):
        x, label = X[i], y[i]

        # Forward
        z = linear.forward(x)[0]           
        y_pred = 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Sigmoid
        loss = -(label * np.log(y_pred + 1e-15) + (1 - label) * np.log(1 - y_pred + 1e-15))
        total_loss += loss

        # Backward
        dL_dz = y_pred - label           
        dx = linear.backward(np.array([dL_dz]))

        # Update
        linear.A -= lr * linear.dA
        linear.b -= lr * linear.db

    losses.append(total_loss / len(X))
    if epoch % 10 == 0:
        print(f"Sample {epoch}, Loss: {losses[-1]:.4f}")
