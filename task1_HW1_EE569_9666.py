import numpy as np

class Linear:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def forward(self, x):
        self.x = x 
        return np.dot(self.A, x) + self.b

    def backward(self, dy):
     
        dx = np.dot(self.A.T, dy)       
        self.dA = np.outer(dy, self.x) 
        self.db = dy  

        return dx

print("Example 1: 2 inputs → 1 output")

# Parameters: A is (1, 2), b is (1,)
A1 = np.array([[1.5, -0.5]])   
b1 = np.array([0.2])          

# Input: x is (2,)
x1 = np.array([2.0, 1.0])

# Create node
linear1 = Linear(A1, b1)

# Forward pass
y1 = linear1.forward(x1)
print("Forward Pass")
print(f"Input x: {x1}")
print(f"Output y = A·x + b = {y1}")

dy1 = np.array([0.8])  

# Backward pass
dx1 = linear1.backward(dy1)
print("Backwards Pass")
print(f"Upstream gradient dy: {dy1}")
print(f"Gradient dx: {dx1}")       
print(f"Gradient dA :\n{linear1.dA}") 
print(f"Gradient  db: {linear1.db}")  
