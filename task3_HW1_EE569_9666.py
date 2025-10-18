import numpy as np

#--- Task 1: Linear Node 
class Linear:
    def __init__(self, A, b):
  
        self.A = np.array(A).reshape(1, -1)  # (1, d)
        self.b = np.array(b).reshape(1,)     # (1,)

    def forward(self, X):
        """
        X: (batch_size, input_dim)
        Returns: (batch_size, 1)
        """
        self.X = X
        return X @ self.A.T + self.b  # broadcasting b

    def backward(self, dY):
        # dA = sum over batch of (dY_i * X_i) → shape (1, d)
        self.dA = (dY.T @ self.X)  # (1, batch) @ (batch, d) → (1, d)
        self.db = np.sum(dY, axis=0)  # (1,)
        dX = dY @ self.A  # (batch, 1) @ (1, d) → (batch, d)
        return dX


#Task 3: Batching by linear regression 
np.random.seed(0)

# random datasets
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)  # (100, 1)


A = np.random.randn(2) * 0.01  # متجه (2,)
b = 0.0
linear = Linear(A, b)

lr = 0.1
batch_size = 8  

losses = []

for epoch in range(50):

    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    total_loss = 0
    num_batches = 0

    for i in range(0, len(X), batch_size):
        X_batch = X_shuffled[i:i+batch_size]      
        y_batch = y_shuffled[i:i+batch_size]      

        # Forward
        Z = linear.forward(X_batch)               
        Y_pred = 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Sigmoid
        loss = -np.mean(y_batch * np.log(Y_pred + 1e-15) + (1 - y_batch) * np.log(1 - Y_pred + 1e-15))
        total_loss += loss
        num_batches += 1

        # Backward
        dZ = Y_pred - y_batch  
        dX = linear.backward(dZ)

        # Update
        linear.A -= lr * linear.dA
        linear.b -= lr * linear.db

    avg_loss = total_loss / num_batches
    losses.append(avg_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
