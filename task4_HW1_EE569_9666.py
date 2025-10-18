import numpy as np
import matplotlib.pyplot as plt



class Linear:
    def __init__(self, A, b):
        self.A = A.copy()
        self.b = b.copy()

    def forward(self, x):
        self.x = x
        return x @ self.A + self.b

    def backward(self, dy):
        self.dA = self.x.T @ dy          
        self.db = np.sum(dy, axis=0)     
        dx = dy @ self.A.T               
        return dx



class Sigmoid:
    def forward(self, z):
        z = np.clip(z, -500, 500)        
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self, dL_dout):
        return dL_dout * self.out * (1 - self.out)


class BinaryCrossEntropy:
    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.mean(loss)

    def backward(self):
        N = self.y_true.shape[0]
        dL_dy = -(self.y_true / self.y_pred) + ((1 - self.y_true) / (1 - self.y_pred))
        return dL_dy / N


def generate_data(n_samples=100, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    return X, y


def train_logistic_regression(X, y, batch_size, epochs=100, lr=1.0, seed=42):
    np.random.seed(seed)
    n_samples, input_dim = X.shape
    output_dim = 1

    # initialize weights
    A = np.random.randn(input_dim, output_dim) * 0.1
    b = np.zeros(output_dim)

    linear = Linear(A, b)
    sigmoid = Sigmoid()
    loss_fn = BinaryCrossEntropy()
    losses = []

    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        epoch_loss = 0
        num_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # forward pass
            z = linear.forward(x_batch)
            y_pred = sigmoid.forward(z).squeeze()
            loss = loss_fn.forward(y_pred, y_batch)
            epoch_loss += loss
            num_batches += 1

            # backward pass
            dL_dy = loss_fn.backward()
            dz = sigmoid.backward(dL_dy[:, None])
            _ = linear.backward(dz)

            # parameter update
            linear.A -= lr * linear.dA
            linear.b -= lr * linear.db

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

    return losses



if __name__ == "__main__":
    # Generate dataset
    X, y = generate_data(100)

    # Define batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 100]

    results = {}
    for bs in batch_sizes:
        print(f"Training with batch size = {bs}")
        losses = train_logistic_regression(X, y, batch_size=bs, epochs=100, lr=1.0)
        results[bs] = losses

    # Plot show
    plt.figure(figsize=(10, 6))
    for bs, losses in results.items():
        plt.plot(losses, label=f'Batch Size = {bs}')
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (Binary Cross-Entropy)")
    plt.title("Effect of Batch Size on Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("batch_size_effect.png", dpi=150)
    plt.show()

    print("\n✅ Plot saved as 'batch_size_effect.png'")
