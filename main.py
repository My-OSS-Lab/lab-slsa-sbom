import torch

# 1. Create a basic tensor (the core data structure in PyTorch)
# This creates a 2x3 matrix filled with random numbers
x = torch.rand(2, 3)
print("Random Tensor:\n", x)

# 2. Perform a mathematical operation
# Multiplying the tensor elements by 2
y = x * 2
print("\nMultiplied Tensor:\n", y)

# 3. Check shape and data type
print(f"\nShape of tensor: {x.shape}")
print(f"Data type of tensor: {x.dtype}")

# 4. Move to GPU (CUDA) if available
if torch.cuda.is_available():
    x = x.to("cuda")
    print("\nTensor moved to GPU successfully!")
else:
    print("\nGPU not available, running on CPU.")

