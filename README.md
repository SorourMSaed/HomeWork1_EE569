# HomeWork1_EE569
This assignment focused on building a logistic regression model from scratch using a computation graph approach, with an emphasis on modular design, gradient computation, and the impact of batching on training dynamics. The implementation was restricted to using only NumPy, Matplotlib, and SciPy, ensuring a deep understanding of the underlying mechanics without relying on high-level frameworks. 
 
Task 1: Implementing the Linear Computation Node 

I designed a Linear class that encapsulates the affine transformation y=Ax+b , where A  is a weight matrix, b  is a bias vector, and x  is the input. The node includes: 

    A forward method that computes the output.
    A backward method that correctly computes gradients with respect to the input x , weights A , and bias b  using the chain rule:
        ∂x∂L​=A⊤⋅∂y∂L​ 
        ∂A∂L​=∂y∂L​⋅x⊤ 
        ∂b∂L​=∂y∂L​ 
         
     

This modular design replaces manual operations and ensures clean, reusable code. 
 
Task 2: Integrating the Linear Node into Logistic Regression 

I replaced the traditional separate Multiply and Add nodes in a logistic regression pipeline with the new Linear node. The model uses: 

    A sigmoid activation for probability output.
    Binary cross-entropy loss for optimization.
    The training loop (using stochastic gradient descent) successfully converged, confirming that the Linear node functions correctly within the full computational graph.
     

 
Task 3: Introducing Batching 

I extended the implementation to support batched inputs, where multiple samples are processed simultaneously. Key changes included: 

    Modifying the Linear node to accept input matrices of shape (batch_size, input_dim).
    Using matrix multiplication (X @ A.T) for efficient forward computation.
    Aggregating gradients over the batch (e.g., summing dY for bias updates).
    This significantly improved computational efficiency and prepared the model for Task 4.
     

 
Task 4: Investigating the Effect of Batch Size 

I trained the model with exponentially increasing batch sizes: 1, 2, 4, 8, 16, 32, 64, and 100 (full dataset). For each, I recorded the training loss over epochs and plotted the results. Key observations: 

    Small batch sizes (1–4): High variance in loss due to noisy gradient estimates, but often better generalization.
    Medium batch sizes (8–32): Balanced trade-off between stability and convergence speed.
    Large/full batch (64–100): Smooth, stable loss curves but required careful learning rate tuning to avoid slow convergence or overshooting.
     

The plots clearly illustrate how batch size influences optimization behavior—a fundamental concept in deep learning. 
 
Significance of the Assignment 

This assignment provided hands-on experience with: 

    The mathematical foundations of backpropagation and gradient computation.and why understanding the low-level details remains essential for debugging, optimization, and innovation.
     

 
    The design of modular, reusable components in machine learning systems.
    The practical impact of hyperparameters like batch size on training dynamics.
    By implementing everything from scratch, I gained a deeper appreciation for how high-level libraries (e.g., PyTorch, TensorFlow) abstract these operations—an
