#!/bin/bash

echo " TensorParallelism Backend Directory Structure:"
echo "-----------------------------------------------"

# Starting from backend/
tree -I "__pycache__|*.o|*.out|*.pt|*.so" --dirsfirst

echo "-----------------------------------------------"
echo " End of directory listing "
