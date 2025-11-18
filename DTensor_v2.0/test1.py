# import dtensor

# dtensor.init()
# print("DTensor bindings initialized ✅")

# print("Available methods:", dir(dtensor))

import dtensor

# Initialize CUDA and symbol linkage
dtensor.init()

# Create ProcessGroup for single GPU
id = dtensor.get_unique_id()
pg = dtensor.ProcessGroup(0, 1, 0, id)

# Construct DTensor safely
a = dtensor.DTensor(0, 1, pg)

# Test data transfer
a.setData([1, 2, 3, 4], [2, 2])
print(a.getData())
