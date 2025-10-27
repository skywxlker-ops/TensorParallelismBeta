#!/usr/bin/env python3
"""
Simple MLP test using our DTensor
"""
import numpy as np
import sys
import os

# Add DTensor to path
sys.path.append('./DTensor')

try:
    from simple_dtensor import DTensor
    print("✓ DTensor imported successfully!")
except ImportError as e:
    print(f"✗ Failed to import DTensor: {e}")
    print("Please build the extension first:")
    print("cd DTensor && python setup.py build_ext --inplace")
    sys.exit(1)


class SimpleMLP:
    """Simple MLP using our DTensor for tensor parallelism"""
    
    def __init__(self, input_size, hidden_size, output_size, world_size):
        self.world_size = world_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Split dimensions across devices
        self.local_hidden = hidden_size // world_size
        self.local_output = output_size // world_size
        
        print(f"Local hidden size per device: {self.local_hidden}")
        print(f"Local output size per device: {self.local_output}")
        
        # Create DTensors for weights and biases
        # Weight1: hidden_size x input_size (split by rows)
        self.weight1_dt = DTensor(world_size, self.local_hidden * input_size)
        
        # Weight2: output_size x hidden_size (split by rows)  
        self.weight2_dt = DTensor(world_size, self.local_output * hidden_size)
        
        # Biases
        self.bias1_dt = DTensor(world_size, self.local_hidden)
        self.bias2_dt = DTensor(world_size, self.local_output)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize weights using Xavier initialization"""
        print("Initializing weights...")
        for rank in range(self.world_size):
            # Weight1 for this rank: (local_hidden x input_size)
            w1 = np.random.randn(self.local_hidden, self.input_size).astype(np.float32)
            w1 *= np.sqrt(2.0 / (self.input_size + self.local_hidden))
            self.weight1_dt.set_slice(rank, w1.flatten())
            print(f"Rank {rank} - Weight1 shape: {w1.shape}")
            
            # Bias1 for this rank: (local_hidden,)
            b1 = np.zeros(self.local_hidden, dtype=np.float32)
            self.bias1_dt.set_slice(rank, b1)
            print(f"Rank {rank} - Bias1 shape: {b1.shape}")
            
            # Weight2 for this rank: (local_output x hidden_size)
            w2 = np.random.randn(self.local_output, self.hidden_size).astype(np.float32)
            w2 *= np.sqrt(2.0 / (self.hidden_size + self.local_output))
            self.weight2_dt.set_slice(rank, w2.flatten())
            print(f"Rank {rank} - Weight2 shape: {w2.shape}")
            
            # Bias2 for this rank: (local_output,)
            b2 = np.zeros(self.local_output, dtype=np.float32)
            self.bias2_dt.set_slice(rank, b2)
            print(f"Rank {rank} - Bias2 shape: {b2.shape}")
    
    def relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def forward(self, x, rank=0):
        """
        Forward pass for a specific rank
        x: input vector of shape (input_size,)
        rank: which device to use
        """
        print(f"\n--- Forward pass on Rank {rank} ---")
        print(f"Input shape: {x.shape}")
        
        # Get local weights and biases for this rank
        w1_flat = self.weight1_dt.get_slice(rank)
        b1 = self.bias1_dt.get_slice(rank)
        w2_flat = self.weight2_dt.get_slice(rank)
        b2 = self.bias2_dt.get_slice(rank)
        
        # Reshape weights
        w1 = w1_flat.reshape(self.local_hidden, self.input_size)
        w2 = w2_flat.reshape(self.local_output, self.hidden_size)
        
        print(f"Reshaped Weight1 shape: {w1.shape}")
        print(f"Bias1 shape: {b1.shape}")
        print(f"Reshaped Weight2 shape: {w2.shape}")
        print(f"Bias2 shape: {b2.shape}")
        
        # Layer 1: (local_hidden x input_size) @ (input_size,) -> (local_hidden,)
        print(f"Layer 1: {w1.shape} @ {x.shape} + {b1.shape}")
        h_local = np.dot(w1, x) + b1
        h_local = self.relu(h_local)
        print(f"Local hidden state shape: {h_local.shape}")
        
        # For demonstration, create a full hidden state by repeating local state
        # In real tensor parallelism, you'd all-gather from all devices here
        full_hidden = np.zeros(self.hidden_size, dtype=np.float32)
        start_idx = rank * self.local_hidden
        end_idx = start_idx + self.local_hidden
        full_hidden[start_idx:end_idx] = h_local
        
        print(f"Full hidden state shape: {full_hidden.shape}")
        
        # Layer 2: (local_output x hidden_size) @ (hidden_size,) -> (local_output,)
        print(f"Layer 2: {w2.shape} @ {full_hidden.shape} + {b2.shape}")
        output_local = np.dot(w2, full_hidden) + b2
        print(f"Local output shape: {output_local.shape}")
        
        return output_local
    
    def print_weights(self):
        """Print weights for debugging"""
        print("\nWeight1 DTensor:")
        self.weight1_dt.print_slices()
        print("\nBias1 DTensor:")
        self.bias1_dt.print_slices()
        print("\nWeight2 DTensor:")
        self.weight2_dt.print_slices()
        print("\nBias2 DTensor:")
        self.bias2_dt.print_slices()


def test_dtensor_basic():
    """Test basic DTensor functionality"""
    print("Testing Basic DTensor Functionality")
    print("=" * 50)
    
    dt = DTensor(2, 4)  # 2 devices, 4 elements per slice
    
    print("Initial slices:")
    dt.print_slices()
    
    # Test get/set
    print("\nTesting get/set operations:")
    slice0 = dt.get_slice(0)
    print(f"Slice 0: {slice0}")
    
    new_data = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    dt.set_slice(0, new_data)
    
    print("After setting slice 0:")
    dt.print_slices()
    
    # Test all-reduce
    print("\nTesting all-reduce:")
    dt.all_reduce()
    dt.print_slices()


def test_mlp():
    """Test the MLP with tensor parallelism"""
    print("\nTesting Simple MLP with DTensor")
    print("=" * 50)
    
    # Model parameters
    input_size = 4
    hidden_size = 6  # Make divisible by world_size
    output_size = 2
    world_size = 2
    
    # Create model
    model = SimpleMLP(input_size, hidden_size, output_size, world_size)
    
    print(f"\nModel architecture:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden size: {hidden_size} (split into {world_size} devices)")
    print(f"  Output size: {output_size} (split into {world_size} devices)")
    
    # Create test input
    x = np.array([0.5, -0.2, 0.8, 0.1], dtype=np.float32)
    print(f"\nInput: {x}")
    
    # Test forward pass on each device
    all_outputs = []
    for rank in range(world_size):
        output = model.forward(x, rank)
        all_outputs.append(output)
        print(f"Rank {rank} output: {output}")
    
    # Combine outputs from all devices
    final_output = np.concatenate(all_outputs)
    print(f"\nFinal combined output: {final_output}")
    print(f"Final output shape: {final_output.shape}")
    
    # Test all-reduce on weight tensors
    print("\n" + "="*50)
    print("Testing collective operations:")
    
    print("\nBefore all-reduce on Weight1:")
    model.weight1_dt.print_slices()
    
    print("\nAfter all-reduce on Weight1:")
    model.weight1_dt.all_reduce()
    model.weight1_dt.print_slices()


if __name__ == "__main__":
    test_dtensor_basic()
    test_mlp()