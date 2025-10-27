# DTensor/test_bindings.py
#!/usr/bin/env python3
"""
Simple test to verify DTensor bindings work
"""
import sys
import os
import numpy as np

try:
    from dtensor import ProcessGroup, DTensor
    print("✓ Successfully imported DTensor bindings")
    
    # Test basic functionality
    print("Testing ProcessGroup...")
    pg = ProcessGroup(0, 2, 0)  # rank, world_size, device
    print(f"  Rank: {pg.get_rank()}, World Size: {pg.get_world_size()}")
    
    print("Testing DTensor...")
    dt = DTensor(2, 4)  # world_size, slice_size
    print("Initial slices:")
    dt.print_slices()
    
    print("✓ All tests passed!")
    
except ImportError as e:
    print(f"✗ Failed to import DTensor: {e}")
    print("Make sure to build the extension first:")
    print("  cd DTensor && python setup.py build_ext --inplace")
except Exception as e:
    print(f"✗ Test failed: {e}")